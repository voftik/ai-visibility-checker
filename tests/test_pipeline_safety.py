import asyncio
import base64
import copy
import hashlib
import json
import tempfile
import unittest
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

from httpx import ASGITransport, AsyncClient, ReadTimeout
from sqlalchemy import delete, func, select, text, update

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
from app.services import openrouter as openrouter_service
from app.services.offer_catalog import (
    OfferCatalog,
    OfferCatalogAdmissionError,
    SourceUnit,
    build_domain_research_payload,
    build_offer_catalog,
    build_offer_clusters,
)
from app.services.offer_identity_policy import (
    OFFER_IDENTITY_MODEL_BATCH_VERSION,
    OFFER_IDENTITY_MODEL_DECISION_VERSION,
    OFFER_IDENTITY_RESULT_VERSION,
    OfferIdentityDecision,
    OfferIdentityModelRole,
    OfferIdentityReasonCode,
    build_offer_identity_contract,
)
from app.services.live_russian_policy import (
    LIVE_RUSSIAN_POLICY_MANIFEST,
    lint_reader_copy_tree,
)
from app.services.panel_coverage import build_panel_metric_coverage_admission
from app.services.openrouter import (
    OUTPUT_ENVELOPE_VERSION,
    WEB_ATTESTATION_VERSION,
    OpenRouterAuditCheckpointError,
    OpenRouterOutputLimitError,
    OpenRouterError,
    OpenRouterPolicyError,
    OpenRouterResponseContractError,
    OutputTokenPolicy,
    PanelModel,
    WebSearchPolicy,
    attest_web_response,
    panel_models,
    web_request_policy,
)
from app.services.report_semantic_gate import (
    CANONICAL_OBSERVATIONAL_MEMORY_LIMITATION,
    REPORT_SEMANTIC_MODEL,
    semantic_provider_payload,
    semantic_provider_request_utf8_bytes,
)
from app.services.analyzer import (
    _require_market_research_usable,
    ANALYZER_REDUCER_HARNESS_VERSION,
    ANALYSIS_MODEL,
    ANNOTATION_SCHEMA,
    ANNOTATION_VERSION,
    ENTITY_CATALOG_CHUNK_VERSION,
    ENTITY_CATALOG_VERSION,
    FINAL_REPORT_VERSION,
    FINAL_CONTEXT_MAX_ANSWERS,
    FINAL_REPORT_AUTHOR_ARTIFACT_KEY,
    FINAL_INPUT_EVIDENCE_SCHEMA,
    FINAL_INPUT_ROOT_SUMMARY_SCHEMA,
    FINAL_REPORT_SCHEMA,
    ILLUSTRATION_CONCEPTS_SCHEMA,
    ILLUSTRATION_GENERATION_VERSION,
    ILLUSTRATION_CONCEPT_MODEL,
    ILLUSTRATION_QA_SCHEMA,
    ILLUSTRATION_QA_VERSION,
    ILLUSTRATION_ROLE_CONCURRENCY,
    LEGACY_PANEL_CONTRACT_VERSION,
    LEGACY_PANEL_EVIDENCE_VERSION,
    LEGACY_MEMORY_OBSERVATION_REASON,
    LEGACY_MEMORY_MODELS,
    LEGACY_PROMPT_SET_VERSIONS,
    PROCESSING_BATCH_CONCURRENCY,
    PROCESSING_MODEL,
    BOUNDED_PANEL_CONTRACT_VERSION,
    PANEL_ATTEMPT_AUDIT_VERSION,
    PANEL_CONTRACT_VERSION,
    PANEL_CORPUS_EXPECTED_CELL_COUNT,
    PANEL_CORPUS_RECEIPT_KEY,
    PANEL_CORPUS_RECEIPT_VERSION,
    PANEL_OUTPUT_POLICY,
    MARKET_RESEARCH_SCHEMA,
    MARKET_RESEARCH_VERSION,
    MARKET_RESEARCH_INPUT_HARNESS_VERSION,
    MARKET_RESEARCH_STRUCTURING_HARNESS_VERSION,
    MARKET_RESEARCH_WEB_HARNESS_VERSION,
    METRICS_VERSION,
    PROMPT_SET_REVIEW_SCHEMA,
    PROMPT_SET_SCHEMA,
    PROMPT_SET_VERSION,
    SITE_PROFILE_SCHEMA,
    _annotate_answers,
    _analyzer_model_input_window,
    _annotation_context_payload,
    _annotation_context_sha256,
    _annotation_split_oversized_record,
    _attribution_owner_aliases,
    _answers_for_catalog,
    _attach_offer_identity_policy,
    _artifact_cache_matches,
    _bounded_semantic_model_evidence_context,
    _build_public_report,
    _classify_site,
    _compute_metrics,
    _deterministic_annotation_warnings,
    _deterministic_entity_catalog_union,
    _deterministic_prompt_fallback,
    _deterministic_site_profile_union,
    _edit_illustration_copy_language,
    _entity_catalog,
    _entity_alias_entries,
    _evidence_contains_complete_alias,
    _evidence_is_literal,
    _ensure_answer_rows,
    _FinalSemanticReviewerUnavailable,
    _expected_corpus_cells,
    _final_corpus_manifest,
    _final_input_deterministic_passthrough,
    _final_input_claim_ledger,
    _final_input_preflight,
    _flatten_final_input_payload,
    _final_model_input_window,
    _normalize_final_evidence_packet,
    _normalize_final_root_summary_packet,
    _final_root_tokens_are_grounded,
    _final_root_fact_refs,
    _final_root_fact_table,
    _final_root_semantic_entries,
    _final_root_parent_node,
    _final_root_node_receipt,
    _verify_final_root_tree,
    _preserve_final_evidence_reduction,
    _prepare_final_model_payload,
    _final_report,
    _final_report_author_candidate,
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
    _illustration_receipt_state,
    _illustration_review_errors,
    _legacy_panel_request_sha256,
    _legacy_panel_run_contract,
    _load_panel_resume_checkpoint,
    _long_response_leaf,
    _long_response_lineage,
    _panel_metric_access,
    _market_research,
    _market_research_evidence_tree,
    _market_research_model_window,
    _market_research_structuring_shards,
    _market_research_web_children,
    _market_research_web_leaf,
    _market_research_web_provider_payload,
    _market_research_web_request_utf8_bytes,
    _market_research_web_scope,
    _validate_market_research_web_leaf,
    _market_research_sufficiency,
    _metric_rows,
    _visibility_slice,
    _normalize_unpaired_memory_illustration,
    _panel_answer_attestation,
    _seal_or_validate_panel_corpus_receipt,
    _validate_panel_corpus_receipt_if_present,
    _bounded_panel_request_sha256,
    _panel_request_sha256,
    _panel_web_policy,
    _persist_prompts,
    _portfolio_entity_is_grounded,
    _portfolio_mention_policy,
    _profile_offer_scope_entities,
    _preserve_entity_catalog_reduction,
    _preserve_site_profile_reduction,
    _prompt_review_errors,
    _processing_artifact,
    _probe_access_outcome,
    _reconcile_annotation,
    _reader_copy_document,
    _reader_copy_gate_decision,
    _reader_copy_publication_contract,
    _refresh_saved_reprocess_source_foundation,
    _recover_prompt_set,
    reprocess_saved_answers,
    _run_panel,
    _rendering_assessment,
    _review_prompt_set_semantics,
    _review_illustration,
    _render_markdown,
    _reuse_saved_illustration_assets,
    _reused_illustration_validation_jobs,
    _saved_illustration_file_path,
    _restrict_partition_annotation_to_core,
    _scope_entity_catalog_to_profile,
    _serialized_llm_request_bytes,
    _structured_artifact,
    _structured_provider_request_utf8_bytes,
    _run_report_branches,
    _run_reused_report_branches,
    _rows_from_full_answer_models,
    _sanitize_optional_report_assets,
    _sanitize_headline_emphasis,
    _semantic_reviewer_failure_is_fail_soft,
    _select_final_answer_context,
    _site_context,
    _stable_json_sha256,
    _unannotated_answers,
    _validate_final_report,
    _validate_illustration_concepts,
    _verified_illustration_asset_receipts,
    _validate_prompt_set,
    analyze_run,
    MarketResearchGateError,
    PanelCheckpointMismatchError,
)
from app.services.publication_contract import (
    OPTIONAL_ASSET_ADMISSION_PREFIX,
    PublicationContractError,
)
from app.services.report_editor import (
    edit_report,
    illustration_copy_narrative_paths,
    seal_editorial_audit,
)
from app.services.content_extractor import extract_text_signals
from app.services.robots_parser import parse_robots, robots_path_allowed
from app.services.recovery_orchestrator import (
    ACTION_DETERMINISTIC_FALLBACK,
    ACTION_RETRY_WITH_GUIDANCE,
    ACTION_STOP,
    RecoveryPlannerUnavailable,
)


def _rules_by_bot(value: str) -> dict[str, tuple[str, str]]:
    return {bot: (rule, raw) for bot, rule, raw in parse_robots(value)}


class LongResponseReducerSafetyTests(unittest.TestCase):
    def test_lineage_rejects_missing_and_duplicate_units(self) -> None:
        left = _long_response_leaf({"value": 1}, ["unit-1"])
        right = _long_response_leaf({"value": 2}, ["unit-2"])

        self.assertEqual(
            _long_response_lineage(
                [left, right],
                expected_unit_ids=["unit-1", "unit-2"],
            ),
            ["unit-1", "unit-2"],
        )
        with self.assertRaisesRegex(OpenRouterError, "missing"):
            _long_response_lineage(
                [left],
                expected_unit_ids=["unit-1", "unit-2"],
            )
        with self.assertRaisesRegex(OpenRouterError, "duplicated"):
            _long_response_lineage([left, left])

    def test_final_reducer_preserves_distinct_supporting_facts_from_same_unit(
        self,
    ) -> None:
        first = {
            "category": "context",
            "statement": "Сайт отвечает HTTP 200.",
            "source_paths": ["/technical/pages/0/status"],
            "source_unit_ids": ["unit-1"],
            "exact_values": ["200"],
            "evidence_excerpt": "200",
            "importance": "supporting",
        }
        second = {
            "category": "context",
            "statement": "Текст страницы доступен в HTML.",
            "source_paths": ["/technical/pages/0/status"],
            "source_unit_ids": ["unit-1"],
            "exact_values": ["server_html"],
            "evidence_excerpt": "server_html",
            "importance": "supporting",
        }
        child = {
            "observations": [first, second],
            "uncertainties": [],
            "report_focus": [],
            "unit_coverage": [
                {
                    "source_unit_id": "unit-1",
                    "disposition": "supporting_context",
                    "rationale": "Оба факта принадлежат одной source unit.",
                }
            ],
        }
        reducer_output = {
            **child,
            "observations": [first],
        }

        preserved = _preserve_final_evidence_reduction(
            reducer_output,
            [child],
        )

        self.assertEqual(preserved["observations"], [first, second])

    def test_annotation_context_fragments_send_overlap_with_core_ownership(
        self,
    ) -> None:
        record = {
            "record_id": "catalog:entity:0",
            "kind": "catalog_entity",
            "field": "entities",
            "value": {
                "canonical_name": ("L" * 250 + " BoundaryEntity " + "R" * 320),
                "aliases": ["BoundaryEntity"],
            },
        }
        serialized = json.dumps(
            record["value"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

        fragments = _annotation_split_oversized_record(
            record,
            target_chars=256,
        )

        self.assertGreater(len(fragments), 1)
        reconstructed = "".join(
            fragment["value"][
                fragment["core_start_in_context"] : fragment["core_end_in_context"]
            ]
            for fragment in fragments
        )
        self.assertEqual(reconstructed, serialized)
        self.assertTrue(
            any(
                fragment["core_start_in_context"] > 0
                or fragment["core_end_in_context"] < len(fragment["value"])
                for fragment in fragments
            )
        )

        payload = _annotation_context_payload(
            profile={"brand_name": "VK"},
            catalog={"entities": [record["value"]]},
            full_alias_map={"boundaryentity": ["VK"]},
            all_records=[record],
            selected_records=[record],
            shard_records=fragments,
            selection_rule="test",
            answers=[],
            research_guidance="",
            repair_mode="",
            shard_index=0,
            shard_count=1,
        )
        sent_fragments = payload["context_fragments"]
        self.assertEqual(
            [item["json_fragment"] for item in sent_fragments],
            [item["value"] for item in fragments],
        )
        self.assertEqual(
            payload["entity_attribution_aliases"],
            {"boundaryentity": ["VK"]},
        )
        self.assertTrue(
            all(
                item["ownership_rule"] == "first_decisive_evidence_character_in_core"
                and item["core_sha256"]
                and item["context_sha256"]
                for item in sent_fragments
            )
        )

    def test_reducer_omissions_become_uncertainties_not_active_entities(self) -> None:
        profile = _preserve_site_profile_reduction(
            {
                "brand_aliases": ["Target"],
                "products": [],
                "entity_scope": [],
                "uncertainties": [],
            },
            [
                {
                    "brand_aliases": ["Target", "Leaf alias"],
                    "products": ["Unconfirmed service"],
                    "entity_scope": [{"canonical_name": "Leaf product"}],
                    "uncertainties": [],
                }
            ],
        )
        self.assertEqual(profile["brand_aliases"], ["Target"])
        self.assertEqual(profile["products"], [])
        self.assertEqual(profile["entity_scope"], [])
        self.assertTrue(
            any("Leaf product" in value for value in profile["uncertainties"])
        )

        catalog = _preserve_entity_catalog_reduction(
            {"target_aliases": ["Target"], "entities": [], "uncertainties": []},
            [
                {
                    "target_aliases": ["Target", "Leaf alias"],
                    "entities": [
                        {
                            "canonical_name": "Unconfirmed product",
                            "aliases": ["Leaf product"],
                        }
                    ],
                    "uncertainties": [],
                }
            ],
        )
        self.assertEqual(catalog["target_aliases"], ["Target"])
        self.assertEqual(catalog["entities"], [])
        self.assertTrue(
            any("Unconfirmed product" in value for value in catalog["uncertainties"])
        )

    def test_overlap_context_does_not_double_own_literal_facts(self) -> None:
        context = "Realweb — владелец услуги.\nCORE: programmatic-реклама"
        core_start = context.index("CORE")
        unit = {
            "answer": context,
            "_lr_core_start_in_context": core_start,
            "_lr_core_end_in_context": len(context),
        }
        annotation = {
            "target_mentioned": True,
            "target_position": 1,
            "target_role": "recommended",
            "sentiment": "positive",
            "entity_mentions": [
                {"canonical_name": "Realweb", "evidence": "Realweb"},
                {
                    "canonical_name": "programmatic-реклама",
                    "evidence": "programmatic-реклама",
                },
            ],
            "evidence": ["Realweb", "programmatic-реклама"],
        }

        restricted = _restrict_partition_annotation_to_core(
            annotation,
            unit,
            target_aliases=["Realweb"],
        )

        self.assertFalse(restricted["target_mentioned"])
        self.assertIsNone(restricted["target_position"])
        self.assertEqual(restricted["sentiment"], "unknown")
        self.assertEqual(
            [item["canonical_name"] for item in restricted["entity_mentions"]],
            ["programmatic-реклама"],
        )
        self.assertEqual(restricted["evidence"], ["programmatic-реклама"])

        inconsistent = _restrict_partition_annotation_to_core(
            {
                **annotation,
                "target_mentioned": False,
                "target_position": 1,
                "target_role": "recommended",
                "sentiment": "negative",
            },
            unit,
            target_aliases=["Realweb"],
        )
        self.assertFalse(inconsistent["target_mentioned"])
        self.assertIsNone(inconsistent["target_position"])
        self.assertEqual(inconsistent["target_role"], "absent")
        self.assertEqual(inconsistent["sentiment"], "unknown")


def _historical_bounded_panel_policy(
    *,
    mode: str,
    provider_key: str,
    model: str,
) -> dict[str, object]:
    policy = _panel_web_policy(mode, provider_key)
    plugin_off = [{"id": "web", "enabled": False}]
    if policy is WebSearchPolicy.REQUIRED:
        mechanism = "openrouter_server_tool"
        request_fields = {
            "plugins": plugin_off,
            "tools": [
                {
                    "type": "openrouter:web_search",
                    "parameters": {
                        "engine": "auto",
                        "max_results": 5,
                        "max_total_results": 12,
                        "max_uses": 3,
                        "search_context_size": "low",
                    },
                }
            ],
            "tool_choice": "auto",
            "max_tool_calls": 4,
        }
    elif policy is WebSearchPolicy.NATIVE_REQUIRED:
        mechanism = "perplexity_native_search"
        request_fields = {"plugins": plugin_off}
    else:
        mechanism = "none"
        request_fields = {
            "plugins": plugin_off,
            "tool_choice": "none",
        }
    contract: dict[str, object] = {
        "version": WEB_ATTESTATION_VERSION,
        "policy": policy.value,
        "mechanism": mechanism,
        "model": model,
        "request_fields": request_fields,
        "requires_url_citation": policy is not WebSearchPolicy.FORBIDDEN,
    }
    return {**contract, "sha256": _stable_json_sha256(contract)}


def _attested_panel_usage(
    *,
    prompt_text: str,
    mode: str,
    provider_key: str,
    model: str,
    response_text: str | None = None,
    contract_version: str = PANEL_CONTRACT_VERSION,
    max_tokens: int | None = None,
) -> tuple[dict[str, object], list[dict[str, str]] | None]:
    policy = _panel_web_policy(mode, provider_key)
    if contract_version == BOUNDED_PANEL_CONTRACT_VERSION:
        request_policy = _historical_bounded_panel_policy(
            mode=mode,
            provider_key=provider_key,
            model=model,
        )
    else:
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
            "web_search_requests": (1 if policy is WebSearchPolicy.REQUIRED else 0)
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
    if contract_version == PANEL_CONTRACT_VERSION:
        request_sha256 = _panel_request_sha256(
            prompt_text=prompt_text,
            mode=mode,
            provider_key=provider_key,
            model=model,
        )
    elif contract_version == BOUNDED_PANEL_CONTRACT_VERSION:
        request_sha256 = _bounded_panel_request_sha256(
            prompt_text=prompt_text,
            mode=mode,
            provider_key=provider_key,
            model=model,
            max_tokens=3_200 if max_tokens is None else max_tokens,
            request_policy=request_policy,
        )
    else:
        raise ValueError(f"Unsupported test panel contract: {contract_version}")
    panel_contract: dict[str, object] = {
        "version": contract_version,
        "request_sha256": request_sha256,
        "request_policy_sha256": request_policy["sha256"],
        "web_policy": request_policy["policy"],
        "attestation_version": WEB_ATTESTATION_VERSION,
        "web_attestation": attestation,
    }
    if contract_version == PANEL_CONTRACT_VERSION:
        panel_contract["output_policy"] = PANEL_OUTPUT_POLICY
        if response_text is not None:
            panel_contract.update(
                {
                    "observation_completeness": "complete",
                    "raw_response_sha256": hashlib.sha256(
                        response_text.encode("utf-8")
                    ).hexdigest(),
                    "raw_response_chars": len(response_text),
                }
            )
    elif max_tokens is not None:
        panel_contract["max_tokens"] = max_tokens

    usage: dict[str, object] = {
        **raw_usage,
        "_aiv_request_policy": request_policy,
        "_aiv_response_annotations": annotations,
        "_aiv_router_metadata": {},
        "_aiv_web_attestation": attestation,
        "_aiv_panel_contract": panel_contract,
    }
    if contract_version == PANEL_CONTRACT_VERSION:
        usage["_aiv_output_envelope"] = {
            "version": OUTPUT_ENVELOPE_VERSION,
            "policy": OutputTokenPolicy.MODEL_MAX.value,
            "requested_model": model,
            "resolution": "test_fixture",
            "context_length": 128_000,
            "max_completion_tokens": 32_000,
        }
        usage["_aiv_transport"] = {
            "output_complete": True,
            "output_limited": False,
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


def _profile_with_offer_contract(
    profile: dict[str, Any],
    *,
    domain: str = "example.com",
    source_url: str | None = None,
    offer_name: str = "Платформа",
    offer_kind: str = "product",
) -> dict[str, Any]:
    """Return a profile fixture with one source-proven commercial offer."""

    enriched = copy.deepcopy(profile)
    brand_name = str(enriched.get("brand_name") or "Example").strip()
    source_url = source_url or f"https://{domain}/"
    source_text = (
        f"{brand_name} предлагает продукт «{offer_name}» для анализа видимости."
    )
    source = SourceUnit.from_text(
        source_unit_id=f"{source_url}:000000",
        source_url=source_url,
        text=source_text,
    )
    jobs = [
        str(value).strip()
        for value in enriched.get("customer_jobs") or []
        if str(value).strip()
    ] or ["Сравнить поставщиков аналитики"]
    catalog = build_offer_catalog(
        client_domain=domain,
        client_aliases=[brand_name, *(enriched.get("brand_aliases") or [])],
        source_units=[source],
        candidates=[
            {
                "canonical_name": offer_name,
                "aliases": [],
                "kind": offer_kind,
                "source_url": source.source_url,
                "evidence_excerpt": source_text,
                "source_unit_id": source.source_unit_id,
                "source_sha256": source.source_sha256,
                "confidence": 0.95,
                "user_jobs": jobs,
                "commercially_relevant": True,
            }
        ],
    )
    clusters = build_offer_clusters(catalog)
    enriched["products"] = list(catalog.legacy_product_strings())
    enriched["offer_catalog"] = catalog.as_dict()
    enriched["offer_clusters"] = [cluster.as_dict() for cluster in clusters]
    enriched["offer_catalog_research_manifest"] = copy.deepcopy(
        build_domain_research_payload(catalog).manifest
    )
    return enriched


def _prompt_set_with_offer_coverage(
    prompts: list[dict[str, Any]],
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Attach the mandatory cluster mapping to a valid prompt-set fixture."""

    cluster_ids = [
        str(item.get("cluster_id") or "")
        for item in profile.get("offer_clusters") or []
        if isinstance(item, dict) and str(item.get("cluster_id") or "")
    ]
    first_discovery = True
    for prompt in prompts:
        if prompt.get("role") == "unbranded_discovery":
            prompt["supporting_cluster_ids"] = (
                list(cluster_ids) if first_discovery else []
            )
            first_discovery = False
        else:
            prompt["supporting_cluster_ids"] = []
    return {"prompts": prompts, "cluster_exclusions": []}


class RobotsParserTests(unittest.TestCase):
    def test_wildcard_group_applies_to_known_bots(self) -> None:
        rules = _rules_by_bot("User-agent: *\nDisallow: /\n")
        self.assertEqual(rules["GPTBot"][0], "disallow_all")
        self.assertEqual(rules["ClaudeBot"][0], "disallow_all")
        self.assertIn("User-agent: *", rules["GPTBot"][1])

    def test_specific_bot_group_overrides_wildcard(self) -> None:
        rules = _rules_by_bot(
            "User-agent: *\nDisallow: /\n\nUser-agent: GPTBot\nAllow: /\n"
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
        self.assertTrue(robots_path_allowed(raw, "https://example.com/private/public"))

    def test_longest_path_rule_and_allow_tie_are_respected(self) -> None:
        raw = "User-agent: *\nDisallow: /catalog/*\nAllow: /catalog/public\n"
        self.assertFalse(robots_path_allowed(raw, "https://example.com/catalog/secret"))
        self.assertTrue(robots_path_allowed(raw, "https://example.com/catalog/public"))


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
        selected = crawler._semantic_frontier_urls(
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
            response_text="Сохранённый ответ.",
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
        self.assertTrue(LEGACY_PANEL_EVIDENCE_VERSION.endswith("evidence-v2"))

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

    def test_required_request_offers_one_server_web_tool(self) -> None:
        fields, _policy = web_request_policy(
            model="openai/gpt-5.4",
            policy=WebSearchPolicy.REQUIRED,
        )

        # Не "required": принуждение держится на каждом витке серверного
        # цикла инструментов и не даёт модели хода на текст. Факт поиска
        # проверяет attest_web_response() уже по ответу.
        self.assertEqual(fields["tool_choice"], "auto")
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

    def test_panel_v3_model_max_envelope_is_attested(self) -> None:
        prompt_text = "Какие решения выбрать?"
        usage, citations = _attested_panel_usage(
            prompt_text=prompt_text,
            mode="web",
            provider_key="openai",
            model="test/model",
            response_text="Ответ.",
        )
        self.assertNotIn("max_tokens", usage["_aiv_panel_contract"])
        self.assertEqual(
            usage["_aiv_panel_contract"]["output_policy"],
            PANEL_OUTPUT_POLICY,
        )
        self.assertEqual(
            usage["_aiv_output_envelope"]["policy"],
            OutputTokenPolicy.MODEL_MAX.value,
        )
        answer = ModelAnswer(
            run_id="run-id",
            prompt_id=1,
            provider_key="openai",
            model="test/model",
            mode="web",
            status="completed",
            response_text="Ответ.",
            citations_json=citations,
            usage_json=usage,
        )

        verified, reason = _panel_answer_attestation(
            answer,
            prompt_text=prompt_text,
        )

        self.assertTrue(verified)
        self.assertEqual(reason, "verified")

    def test_panel_v2_budget_allowlist_and_hash_are_attested(self) -> None:
        prompt_text = "Какие решения выбрать?"
        model = "openai/gpt-chat-latest"
        usage, citations = _attested_panel_usage(
            prompt_text=prompt_text,
            mode="web",
            provider_key="openai",
            model=model,
            contract_version=BOUNDED_PANEL_CONTRACT_VERSION,
            max_tokens=6_400,
        )
        answer = ModelAnswer(
            run_id="run-id",
            prompt_id=1,
            provider_key="openai",
            model=model,
            mode="web",
            status="completed",
            response_text="Ответ.",
            citations_json=citations,
            usage_json=usage,
        )

        verified, reason = _panel_answer_attestation(
            answer,
            prompt_text=prompt_text,
        )
        self.assertTrue(verified)
        self.assertEqual(reason, "verified")

        unapproved_usage = copy.deepcopy(usage)
        unapproved = dict(unapproved_usage["_aiv_panel_contract"])
        unapproved["max_tokens"] = 4_800
        unapproved["request_sha256"] = _bounded_panel_request_sha256(
            prompt_text=prompt_text,
            mode="web",
            provider_key="openai",
            model=model,
            max_tokens=4_800,
            request_policy=unapproved_usage["_aiv_request_policy"],
        )
        unapproved_usage["_aiv_panel_contract"] = unapproved
        answer.usage_json = unapproved_usage
        verified, reason = _panel_answer_attestation(
            answer,
            prompt_text=prompt_text,
        )
        self.assertFalse(verified)
        self.assertEqual(reason, "unapproved_panel_max_tokens")

        tampered_usage = copy.deepcopy(usage)
        tampered = dict(tampered_usage["_aiv_panel_contract"])
        tampered["max_tokens"] = 3_200
        tampered_usage["_aiv_panel_contract"] = tampered
        answer.usage_json = tampered_usage
        verified, reason = _panel_answer_attestation(
            answer,
            prompt_text=prompt_text,
        )
        self.assertFalse(verified)
        self.assertEqual(reason, "request_hash_mismatch")

    def test_panel_v2_rich_web_policy_survives_current_policy_drift(self) -> None:
        prompt_text = "Какие решения выбрать?"
        model = "openai/gpt-chat-latest"
        usage, citations = _attested_panel_usage(
            prompt_text=prompt_text,
            mode="web",
            provider_key="openai",
            model=model,
            contract_version=BOUNDED_PANEL_CONTRACT_VERSION,
            max_tokens=3_200,
        )
        answer = ModelAnswer(
            run_id="historical-run",
            prompt_id=1,
            provider_key="openai",
            model=model,
            mode="web",
            status="completed",
            response_text="Ответ.",
            citations_json=citations,
            usage_json=usage,
        )

        with patch(
            "app.services.analyzer.web_request_policy",
            side_effect=AssertionError("panel-v2 must not use current policy"),
        ):
            verified, reason = _panel_answer_attestation(
                answer,
                prompt_text=prompt_text,
            )

        self.assertTrue(verified)
        self.assertEqual(reason, "verified")

    def test_panel_v2_policy_and_attestation_tampering_fail_closed(self) -> None:
        prompt_text = "Какие решения выбрать?"
        model = "openai/gpt-chat-latest"
        usage, citations = _attested_panel_usage(
            prompt_text=prompt_text,
            mode="web",
            provider_key="openai",
            model=model,
            contract_version=BOUNDED_PANEL_CONTRACT_VERSION,
            max_tokens=3_200,
        )

        def verify(candidate: dict[str, object]) -> tuple[bool, str]:
            answer = ModelAnswer(
                run_id="historical-run",
                prompt_id=1,
                provider_key="openai",
                model=model,
                mode="web",
                status="completed",
                response_text="Ответ.",
                citations_json=citations,
                usage_json=candidate,
            )
            return _panel_answer_attestation(answer, prompt_text=prompt_text)

        def reseal_policy(candidate: dict[str, object]) -> None:
            request_policy = candidate["_aiv_request_policy"]
            assert isinstance(request_policy, dict)
            policy_contract = {
                key: value for key, value in request_policy.items() if key != "sha256"
            }
            request_policy["sha256"] = _stable_json_sha256(policy_contract)
            provenance = candidate["_aiv_panel_contract"]
            assert isinstance(provenance, dict)
            provenance["request_policy_sha256"] = request_policy["sha256"]
            provenance["request_sha256"] = _bounded_panel_request_sha256(
                prompt_text=prompt_text,
                mode="web",
                provider_key="openai",
                model=model,
                max_tokens=3_200,
                request_policy=request_policy,
            )

        corrupt_hash = copy.deepcopy(usage)
        corrupt_hash["_aiv_request_policy"]["sha256"] = "0" * 64
        self.assertEqual(verify(corrupt_hash), (False, "corrupt_request_policy_hash"))

        wrong_policy = copy.deepcopy(usage)
        wrong_policy["_aiv_request_policy"]["policy"] = "forbidden"
        reseal_policy(wrong_policy)
        self.assertEqual(verify(wrong_policy), (False, "request_policy_mismatch"))

        wrong_model = copy.deepcopy(usage)
        wrong_model["_aiv_request_policy"]["model"] = "anthropic/claude-sonnet-5"
        reseal_policy(wrong_model)
        self.assertEqual(
            verify(wrong_model),
            (False, "bounded_request_policy_model_mismatch"),
        )

        wrong_tool = copy.deepcopy(usage)
        wrong_tool["_aiv_request_policy"]["request_fields"]["tools"][0]["parameters"][
            "max_results"
        ] = 6
        reseal_policy(wrong_tool)
        self.assertEqual(
            verify(wrong_tool),
            (False, "bounded_request_policy_fields_mismatch"),
        )

        missing_search = copy.deepcopy(usage)
        missing_search["_aiv_web_attestation"]["web_search_requests"] = 0
        missing_search["_aiv_panel_contract"]["web_attestation"] = copy.deepcopy(
            missing_search["_aiv_web_attestation"]
        )
        self.assertEqual(verify(missing_search), (False, "web_search_not_observed"))

        mismatched_attestation = copy.deepcopy(usage)
        detached_attestation = copy.deepcopy(
            mismatched_attestation["_aiv_web_attestation"]
        )
        detached_attestation["evidence"] = []
        mismatched_attestation["_aiv_web_attestation"] = detached_attestation
        self.assertEqual(
            verify(mismatched_attestation),
            (False, "attestation_provenance_mismatch"),
        )

    def test_panel_v3_does_not_accept_the_historical_rich_policy(self) -> None:
        prompt_text = "Какие решения выбрать?"
        model = "openai/gpt-chat-latest"
        usage, citations = _attested_panel_usage(
            prompt_text=prompt_text,
            mode="web",
            provider_key="openai",
            model=model,
            response_text="Ответ.",
        )
        historical_policy = _historical_bounded_panel_policy(
            mode="web",
            provider_key="openai",
            model=model,
        )
        usage["_aiv_request_policy"] = historical_policy
        usage["_aiv_panel_contract"]["request_policy_sha256"] = historical_policy[
            "sha256"
        ]
        answer = ModelAnswer(
            run_id="current-run",
            prompt_id=1,
            provider_key="openai",
            model=model,
            mode="web",
            status="completed",
            response_text="Ответ.",
            citations_json=citations,
            usage_json=usage,
        )

        verified, reason = _panel_answer_attestation(
            answer,
            prompt_text=prompt_text,
        )

        self.assertFalse(verified)
        self.assertEqual(reason, "request_policy_contract_mismatch")

    def test_historical_panel_v2_keeps_all_81_cells_metric_eligible(self) -> None:
        lanes = [
            ("openai", "web", "openai/gpt-chat-latest"),
            ("gemini", "web", "google/gemini-3.6-flash"),
            ("perplexity", "web", "perplexity/sonar-pro-search"),
            ("deepseek", "web", "deepseek/deepseek-v4-pro"),
            ("claude", "web", "anthropic/claude-sonnet-5"),
            ("openai", "memory", "openai/gpt-chat-latest"),
            ("gemini", "memory", "google/gemini-3.6-flash"),
            ("deepseek", "memory", "deepseek/deepseek-v4-pro"),
            ("claude", "memory", "anthropic/claude-sonnet-5"),
        ]
        expected_cells: list[dict[str, object]] = []
        observed_rows: list[dict[str, object]] = []
        for prompt_id in range(1, 10):
            prompt_text = f"Сценарий {prompt_id}"
            for provider_key, mode, model in lanes:
                usage, citations = _attested_panel_usage(
                    prompt_text=prompt_text,
                    mode=mode,
                    provider_key=provider_key,
                    model=model,
                    contract_version=BOUNDED_PANEL_CONTRACT_VERSION,
                    max_tokens=3_200,
                )
                answer = ModelAnswer(
                    run_id="historical-run",
                    prompt_id=prompt_id,
                    provider_key=provider_key,
                    model=model,
                    mode=mode,
                    status="completed",
                    response_text="Ответ.",
                    citations_json=citations,
                    usage_json=usage,
                )
                verified, reason = _panel_answer_attestation(
                    answer,
                    prompt_text=prompt_text,
                )
                self.assertTrue(verified, reason)
                expected_cells.append(
                    {
                        "prompt_id": prompt_id,
                        "provider_key": provider_key,
                        "mode": mode,
                        "model": model,
                    }
                )
                observed_rows.append(
                    {
                        "prompt_id": prompt_id,
                        "provider_key": provider_key,
                        "mode": mode,
                        "model": model,
                        "status": "completed",
                        "metric_eligible": verified,
                        "metric_evidence_state": "strict_verified",
                        "metric_limitation": None,
                    }
                )

        admission = build_panel_metric_coverage_admission(
            expected_cells=expected_cells,
            observed_rows=observed_rows,
        )
        self.assertEqual(len(observed_rows), PANEL_CORPUS_EXPECTED_CELL_COUNT)
        self.assertTrue(admission["allowed"])
        self.assertEqual(admission["eligible_cell_count"], 81)
        self.assertEqual(admission["coverage_rate"], 1.0)

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
            "brand_knowledge": {"memory": {"data_state": "unavailable"}},
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
        self.assertTrue(_illustration_cache_matches(illustration, "concept-v2"))
        self.assertFalse(_illustration_cache_matches(illustration, "concept-v3"))

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
        self.assertEqual(
            normalized["fact_contract"]["competitors"][0]["name"], "Alternative"
        )
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
        profile = _profile_with_offer_contract(
            {"brand_name": "RW+", "brand_aliases": ["Realweb Plus"]},
            domain="rw.plus",
            source_url="https://rw.plus/",
        )
        prompt_set = _prompt_set_with_offer_coverage(prompts, profile)

        self.assertEqual(_validate_prompt_set(prompt_set, profile), [])

        prompts[2]["text"] = (
            "Кого пригласить в тендер, чтобы получить сильное предложение?"
        )
        self.assertEqual(_validate_prompt_set(prompt_set, profile), [])

        prompts[-1]["text"] = "Что известно об этом поставщике?"
        errors = _validate_prompt_set(prompt_set, profile)
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
        need_based = next(check for check in checks if check["declared_intent"] == "NB")
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
        check_properties = PROMPT_SET_REVIEW_SCHEMA["properties"]["checks"]["items"][
            "properties"
        ]
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
            "pages": [
                {
                    "url": "https://example.com/",
                    "main_text": "Example — аналитическая платформа.",
                }
            ],
        }
        profile = {
            "brand_name": "Example",
            "brand_aliases": [],
            "site_type": "platform",
            "category": "analytics",
            "topics": [],
            "market": "",
            "business_model": "",
            "products": [],
            "audiences": [],
            "customer_jobs": [],
            "decision_criteria": [],
            "geography": [],
            "language": "ru",
            "positioning": "",
            "entity_scope": [],
            "evidence": ["Example"],
            "uncertainties": [],
            "confidence": "medium",
        }

        models: list[str] = []

        async def fake_structured(_run_id: str, **kwargs: Any) -> dict[str, Any]:
            models.append(str(kwargs["model"]))
            payload = kwargs["user_payload"]
            if "page_unit" in payload:
                claim = payload["core_claim"]
                return {
                    "profile": profile,
                    "core_disposition": {
                        "claim_id": claim["claim_id"],
                        "unit_id": claim["unit_id"],
                        "core_sha256": claim["core_sha256"],
                        "disposition": "grounded_fact",
                        "evidence_quote": "Example",
                        "reason": "В core буквально назван бренд Example.",
                    },
                }
            return profile

        with (
            patch(
                "app.services.analyzer._structured_artifact",
                new=fake_structured,
            ),
            patch(
                "app.services.analyzer._analyzer_model_input_window",
                new=AsyncMock(
                    return_value={
                        "input_utf8_window": 200_000,
                        "model_envelope": {
                            "context_length": 200_000,
                            "max_completion_tokens": 20_000,
                        },
                    }
                ),
            ),
        ):
            result = await _classify_site("run-id", context)

        self.assertEqual(result["brand_name"], "Example")
        self.assertEqual(models, [PROCESSING_MODEL, ANALYSIS_MODEL])

    async def test_structured_artifact_preserves_contract_failure_evidence(
        self,
    ) -> None:
        usage = {
            "_aiv_transport": {
                "status": "succeeded",
                "output_complete": False,
                "output_incomplete_reason": "finish_reason:content_filter",
            }
        }
        contract_error = OpenRouterResponseContractError(
            "Structured response was filtered",
            result=SimpleNamespace(
                text='{"partial":',
                usage=usage,
            ),
        )
        with (
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
                "app.services.analyzer.chat_continuable_structured",
                new_callable=AsyncMock,
                side_effect=contract_error,
            ),
            self.assertRaises(OpenRouterResponseContractError),
        ):
            await _structured_artifact(
                "run-id",
                stage_key="scenario_design",
                artifact_key="contract_failure",
                schema={"type": "object"},
                schema_name="contract_failure",
                system="Верни JSON.",
                user_payload={"value": 1},
            )

        failed = save_artifact.await_args_list[-1].kwargs
        running = save_artifact.await_args_list[0].kwargs
        self.assertEqual(running["status"], "running")
        self.assertIs(running["preserve_existing_evidence"], True)
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["raw_text"], '{"partial":')
        self.assertEqual(failed["usage_json"], usage)

    async def test_site_profile_explicitly_models_market_and_customer_choice(
        self,
    ) -> None:
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
        profile = _profile_with_offer_contract(profile)
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
        unit_id = "https://example.com/:000000"
        evidence_tree = {
            "packet": {
                "findings": [
                    {
                        "dimension": "site_confirmed",
                        "claim": "Сайт представляет платформу Example.",
                        "evidence": "Example — аналитическая платформа.",
                        "source_urls": ["https://example.com/"],
                        "source_unit_ids": [unit_id],
                        "confidence": "high",
                    }
                ],
                "uncertainties": [],
                "unit_coverage": [
                    {
                        "source_unit_id": unit_id,
                        "state": "evidence",
                        "note": "Фрагмент прочитан.",
                    }
                ],
                "_aiv_source_unit_ids": [unit_id],
            },
            "manifest": {
                "window": {"input_utf8_window": 96_000},
                "source_unit_count": 1,
                "source_unit_ids": [unit_id],
                "source_unit_ids_sha256": "a" * 64,
            },
            "citations": ready["web_evidence"]["citations"],
            "web_attestation": ready["web_evidence"]["attestation"],
            "retrieval_leaf_count": 1,
            "reducer_levels": 0,
        }
        with (
            patch(
                "app.services.analyzer._cached_market_research",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "app.services.analyzer._market_research_evidence_tree",
                new_callable=AsyncMock,
                return_value=evidence_tree,
            ) as tree_mock,
            patch(
                "app.services.analyzer._final_model_input_window",
                new_callable=AsyncMock,
                return_value={
                    "input_utf8_window": 1_000_000,
                    "model_envelope": {
                        "context_length": 1_000_000,
                        "max_completion_tokens": 100_000,
                    },
                    "resolution": "test",
                },
            ),
            patch(
                "app.services.analyzer.chat_continuable_structured",
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
        tree_payload = tree_mock.await_args.kwargs["payload"]
        self.assertEqual(tree_payload["site_evidence"], site_context["pages"])
        self.assertEqual(chat_mock.await_count, 1)
        structuring_request = chat_mock.await_args.kwargs
        self.assertNotIn("max_tokens", structuring_request)
        self.assertNotIn("max_completion_tokens", structuring_request)
        self.assertIsNotNone(structuring_request.get("response_schema"))
        self.assertTrue(callable(structuring_request.get("audit_checkpoint")))
        self.assertIn("resume_checkpoint", structuring_request)
        structuring_payload = json.loads(structuring_request["messages"][1]["content"])
        self.assertIn("evidence_tree", structuring_payload)
        self.assertNotIn("site_evidence", structuring_payload["site_input"])
        self.assertEqual(
            structuring_payload["evidence_tree_contract"]["source_unit_ids"],
            [unit_id],
        )
        self.assertEqual(
            structuring_payload["confirmed_source_urls"],
            sorted(
                item["url"]
                for item in _ready_market_research()["web_evidence"]["citations"]
            ),
        )
        final_write = save_artifact.await_args.kwargs
        self.assertEqual(final_write["artifact_key"], "market_research")
        self.assertEqual(final_write["prompt_version"], MARKET_RESEARCH_VERSION)
        self.assertEqual(final_write["status"], "completed")
        self.assertEqual(
            final_write["output_json"]["site_confirmed"]["primary_brand"],
            profile["brand_name"],
        )

    async def test_oversized_market_structuring_uses_schema_valid_shards(
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
        profile = _profile_with_offer_contract(profile)
        site_context = {
            "requested_site": {
                "domain": "example.com",
                "url": "https://example.com/",
            },
            "pages": [
                {
                    "url": "https://example.com/",
                    "title": "Example",
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
        unit_id = "https://example.com/:000000"
        tail = "WHOLE-MARKET-STRUCTURING-TAIL"
        evidence_tree = {
            "packet": {
                "findings": [
                    {
                        "dimension": "market",
                        "claim": "Рынок аналитики.",
                        "evidence": ("длинное доказательство " * 2_000) + tail,
                        "source_urls": ["https://research.example/market"],
                        "source_unit_ids": [unit_id],
                        "confidence": "medium",
                    }
                ],
                "uncertainties": [],
                "unit_coverage": [
                    {
                        "source_unit_id": unit_id,
                        "state": "evidence",
                        "note": "Прочитано.",
                    }
                ],
                "_aiv_source_unit_ids": [unit_id],
            },
            "manifest": {
                "source_unit_count": 1,
                "source_unit_ids": [unit_id],
                "source_unit_ids_sha256": "a" * 64,
            },
            "citations": ready["web_evidence"]["citations"],
            "web_attestation": ready["web_evidence"]["attestation"],
            "retrieval_leaf_count": 1,
            "reducer_levels": 0,
        }
        sharded_plan = {
            "version": MARKET_RESEARCH_STRUCTURING_HARNESS_VERSION,
            "mode": "bounded_schema_valid_shards",
            "coverage_complete": True,
        }
        with (
            patch(
                "app.services.analyzer._cached_market_research",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "app.services.analyzer._market_research_evidence_tree",
                new_callable=AsyncMock,
                return_value=evidence_tree,
            ),
            patch(
                "app.services.analyzer._final_model_input_window",
                new_callable=AsyncMock,
                return_value={
                    "input_utf8_window": 128,
                    "model_envelope": {
                        "context_length": 128,
                        "max_completion_tokens": 64,
                    },
                    "resolution": "test",
                },
            ),
            patch(
                "app.services.analyzer._market_research_structuring_shards",
                new_callable=AsyncMock,
                return_value=(
                    parsed,
                    sharded_plan,
                    json.dumps(parsed, ensure_ascii=False),
                    {"_aiv_market_structuring_shards": sharded_plan},
                ),
            ) as shard_mock,
            patch(
                "app.services.analyzer.chat_continuable_structured",
                new_callable=AsyncMock,
            ) as whole_document_chat,
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ),
        ):
            result = await _market_research(
                "run-id",
                profile,
                site_context,
            )

        whole_document_chat.assert_not_awaited()
        shard_mock.assert_awaited_once()
        sharded_source = shard_mock.await_args.kwargs["structuring_payload"]
        self.assertIn(
            tail,
            sharded_source["evidence_tree"]["findings"][0]["evidence"],
        )
        self.assertEqual(result["status"], "ready")

    async def test_market_research_losslessly_maps_huge_pages_and_notes(
        self,
    ) -> None:
        profile = {
            "brand_name": "Huge",
            "evidence": ["Подтверждено сайтом"],
            "customer_jobs": ["Выбрать поставщика"],
            "decision_criteria": ["Надёжность"],
        }
        pages = [
            {
                "url": f"https://huge.example/page-{index}",
                "title": f"Страница {index}",
                "meta_description": "Большая страница",
                "main_text": (f"unit-{index}-evidence " * 1_200),
                "content_storage_state": ("prefix_only" if index == 2 else "complete"),
                "absence_claims_allowed": index != 2,
            }
            for index in range(3)
        ]
        payload = {
            "requested_site": {
                "domain": "huge.example",
                "url": "https://huge.example/",
            },
            "site_profile": profile,
            "site_evidence": pages,
        }
        attestation = {
            "version": WEB_ATTESTATION_VERSION,
            "policy": WebSearchPolicy.REQUIRED.value,
            "state": "verified",
            "metric_eligible": True,
            "web_search_requests": 1,
            "violations": [],
        }
        web_payload_sizes: list[int] = []
        expected_research_notes = ""
        mapped_note_payloads: list[dict[str, Any]] = []
        structured_stage_calls: list[dict[str, Any]] = []

        async def fake_chat(**kwargs: Any) -> SimpleNamespace:
            nonlocal expected_research_notes
            request = json.loads(kwargs["messages"][1]["content"])
            web_payload_sizes.append(
                len(kwargs["messages"][1]["content"].encode("utf-8"))
            )
            unit_id = request["unit_contract"]["source_unit_id"]
            self.assertEqual(unit_id, "market-domain:huge.example")
            citation_url = "https://research.example/market"
            notes = f"Подтверждённый вывод для {unit_id}. Источник {citation_url}. " + (
                "Большие исследовательские заметки. " * 4_500
            )
            expected_research_notes = notes
            return SimpleNamespace(
                parsed=None,
                text=notes,
                usage={},
                web_attestation=copy.deepcopy(attestation),
                citations=[{"url": citation_url, "title": "Источник"}],
            )

        async def fake_structured_artifact(
            *_args: Any,
            **kwargs: Any,
        ) -> dict[str, Any]:
            structured_stage_calls.append(copy.deepcopy(kwargs))
            user_payload = kwargs["user_payload"]
            artifact_key = kwargs["artifact_key"]
            if artifact_key.startswith("market_research_map_"):
                mapped_note_payloads.append(copy.deepcopy(user_payload))
                source = user_payload["source_unit"]
                unit_id = source["source_unit_id"]
                page_url = source["page_url"]
                external_urls = user_payload["confirmed_external_urls"]
                state = (
                    "unknown"
                    if source["content_storage_state"] == "prefix_only"
                    else "evidence"
                )
                findings = [
                    {
                        "dimension": "site_confirmed",
                        "claim": f"Положительный факт {unit_id}",
                        "evidence": "Буквальный фрагмент.",
                        "source_urls": [page_url],
                        "source_unit_ids": [unit_id],
                        "confidence": "high",
                    },
                ]
                if external_urls:
                    findings.append(
                        {
                            "dimension": "market",
                            "claim": f"Рыночный факт {unit_id}",
                            "evidence": "Подтверждено внешним источником.",
                            "source_urls": [external_urls[0]],
                            "source_unit_ids": [unit_id],
                            "confidence": "medium",
                        }
                    )
                return {
                    "findings": findings,
                    "uncertainties": (
                        [
                            {
                                "statement": "Хвост prefix-only неизвестен.",
                                "source_unit_ids": [unit_id],
                            }
                        ]
                        if state == "unknown"
                        else []
                    ),
                    "unit_coverage": [
                        {
                            "source_unit_id": unit_id,
                            "state": state,
                            "note": "Прочитан переданный unit.",
                        }
                    ],
                }
            packets = user_payload["evidence_packets"]
            findings = [
                copy.deepcopy(finding)
                for packet in packets
                for finding in packet["findings"]
            ]
            uncertainties = [
                copy.deepcopy(item)
                for packet in packets
                for item in packet["uncertainties"]
            ]
            coverage = [
                copy.deepcopy(item)
                for packet in packets
                for item in packet["unit_coverage"]
            ]
            return {
                "findings": findings,
                "uncertainties": uncertainties,
                "unit_coverage": coverage,
            }

        with (
            patch(
                "app.services.analyzer.model_output_envelope",
                new_callable=AsyncMock,
                return_value={
                    "context_length": 40_000,
                    "max_completion_tokens": 8_000,
                },
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
                side_effect=fake_chat,
            ) as chat_mock,
            patch(
                "app.services.analyzer._structured_artifact",
                new_callable=AsyncMock,
                side_effect=fake_structured_artifact,
            ),
            patch(
                "app.services.analyzer.asyncio.timeout",
                side_effect=AssertionError(
                    "a progressing retrieval tree must not have an "
                    "aggregate wall-clock timeout"
                ),
            ) as aggregate_timeout,
        ):
            tree = await _market_research_evidence_tree(
                "run-id",
                payload=payload,
                research_system="Исследуй один lossless unit.",
            )

        manifest = tree["manifest"]
        self.assertEqual(manifest["version"], MARKET_RESEARCH_INPUT_HARNESS_VERSION)
        self.assertGreater(manifest["source_unit_count"], len(pages))
        self.assertEqual(chat_mock.await_count, 1)
        self.assertEqual(manifest["web_retrieval_tree_count"], 1)
        self.assertEqual(
            manifest["web_retrieval_contract"]["retrieval_granularity"],
            "one_domain_identity_tree",
        )
        aggregate_timeout.assert_not_called()
        retrieval_contract = manifest["web_retrieval_contract"]
        self.assertIsNone(retrieval_contract["aggregate_liveness_deadline_seconds"])
        self.assertNotIn("liveness_deadline_seconds", retrieval_contract)
        self.assertEqual(
            retrieval_contract["liveness_policy"],
            "per_post_inactivity_and_cancellation_with_durable_resume",
        )
        self.assertEqual(
            retrieval_contract["progress_guards"],
            [
                "exact_task_identity",
                "finite_predeclared_facets",
                "ancestor_raw_sha256_no_repeat",
                "bounded_semantic_retry_tree",
            ],
        )
        self.assertEqual(retrieval_contract["semantic_retry_post_cap"], 96)
        self.assertEqual(retrieval_contract["facet_refinement_depth_cap"], 1)
        self.assertEqual(
            sum(item["source_chars"] for item in manifest["source_manifests"]),
            sum(len(page["main_text"]) for page in pages),
        )
        self.assertEqual(
            tree["packet"]["_aiv_source_unit_ids"],
            sorted(manifest["source_unit_ids"]),
        )
        self.assertGreater(
            len(mapped_note_payloads),
            manifest["source_unit_count"],
        )
        note_payloads_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in mapped_note_payloads:
            note_payloads_by_source[str(item["source_unit"]["source_unit_id"])].append(
                item
            )
        nonempty_note_sources = [
            unit_id
            for unit_id, fragments in note_payloads_by_source.items()
            if any(fragment["research_notes"] for fragment in fragments)
        ]
        self.assertEqual(
            nonempty_note_sources,
            [manifest["external_notes_joined_source_unit_id"]],
        )
        for unit_id in nonempty_note_sources:
            fragments = sorted(
                note_payloads_by_source[unit_id],
                key=lambda item: int(item["research_notes_contract"]["unit_index"]),
            )
            reconstructed = "".join(
                item["research_notes"][
                    int(item["research_notes_contract"]["core_start_in_context"]) : int(
                        item["research_notes_contract"]["core_end_in_context"]
                    )
                ]
                for item in fragments
            )
            self.assertEqual(reconstructed, expected_research_notes)
            self.assertTrue(
                all(
                    item["research_notes_contract"]["ownership_rule"]
                    == "first_decisive_evidence_character_in_core"
                    for item in fragments
                )
            )
        self.assertLessEqual(
            max(web_payload_sizes),
            manifest["window"]["input_utf8_window"],
        )
        self.assertEqual(
            {item["url"] for item in tree["citations"]},
            {"https://research.example/market"},
        )
        self.assertTrue(
            any(item["state"] == "unknown" for item in tree["packet"]["unit_coverage"])
        )
        self.assertTrue(structured_stage_calls)
        self.assertTrue(
            all(call.get("continuable") is True for call in structured_stage_calls)
        )
        self.assertTrue(
            all(
                str(call.get("artifact_key") or "").startswith(
                    ("market_research_map_", "market_research_reduce_")
                )
                for call in structured_stage_calls
            )
        )
        attempt_writes = [
            call.kwargs
            for call in save_artifact.await_args_list
            if call.kwargs["artifact_key"].startswith("market_research_web_attempt_")
            and call.kwargs["status"] == "completed"
        ]
        self.assertEqual(len(attempt_writes), manifest["web_retrieval_post_count"])
        self.assertTrue(all(call["raw_text"] for call in attempt_writes))

    async def test_market_web_limit_refines_without_losing_unicode_prefix(
        self,
    ) -> None:
        source_unit = {
            "url": "https://unicode.example/",
            "title": "Очень длинный рынок 🌍",
            "meta_description": "Описание",
            "main_text": "Контекст сайта",
            "_lr_unit_id": "https://unicode.example/:000000",
            "_lr_unit_index": 0,
            "_lr_unit_count": 1,
            "_lr_unit_sha256": "a" * 64,
            "_lr_context_sha256": "b" * 64,
            "_lr_start_char": 0,
            "_lr_end_char": 14,
        }
        attestation = {
            "version": WEB_ATTESTATION_VERSION,
            "policy": WebSearchPolicy.REQUIRED.value,
            "state": "verified",
            "metric_eligible": True,
            "web_search_requests": 1,
            "violations": [],
        }
        root_prefix = "Начало 🌍漢字🙂 " * 20_000
        raw_by_path = {
            "root": root_prefix,
            "root.0": "Рынок, темы и география подтверждены источником.",
            "root.1": "Аудитории, задачи, критерии и термины подтверждены.",
        }
        seen_payloads: list[dict[str, Any]] = []

        def result_for(path: str, *, limited: bool) -> SimpleNamespace:
            citation = {
                "url": f"https://research.example/{path}",
                "title": path,
                "content": "Подтверждение",
            }
            return SimpleNamespace(
                text=raw_by_path[path],
                usage={},
                web_attestation=copy.deepcopy(attestation),
                citations=[citation],
                transport={
                    "output_limited": limited,
                    "output_complete": not limited,
                },
            )

        async def fake_chat(**kwargs: Any) -> SimpleNamespace:
            self.assertFalse(kwargs["accept_output_limited"])
            payload = json.loads(kwargs["messages"][1]["content"])
            seen_payloads.append(payload)
            path = payload["research_scope"]["path"]
            result = result_for(path, limited=path == "root")
            if path == "root":
                raise OpenRouterOutputLimitError(
                    "provider output limit",
                    result=result,
                )
            return result

        model_envelope = {
            "context_length": 1_000_000,
            "max_completion_tokens": 100_000,
        }
        window_bytes = 800_000
        with (
            patch(
                "app.services.analyzer._artifact_output",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ),
            patch(
                "app.services.analyzer.chat",
                new_callable=AsyncMock,
                side_effect=fake_chat,
            ) as chat_mock,
        ):
            output = await _market_research_web_leaf(
                "run-id",
                source_unit=source_unit,
                requested_site={"domain": "unicode.example"},
                site_profile={"brand_name": "Unicode"},
                system="Исследуй рынок и цитируй источники.",
                window_bytes=window_bytes,
                model_envelope=model_envelope,
            )

        self.assertEqual(chat_mock.await_count, 3)
        self.assertEqual(
            output["research_notes"],
            "\n\n".join(raw_by_path[path] for path in ("root", "root.0", "root.1")),
        )
        self.assertEqual(
            output["observation_completeness"],
            "scoped_composable_positive_evidence",
        )
        manifest = output["retrieval_manifest"]
        self.assertEqual(manifest["version"], MARKET_RESEARCH_WEB_HARNESS_VERSION)
        self.assertEqual(manifest["node_count"], 3)
        self.assertEqual(output["web_attestation"]["leaf_count"], 3)
        for row, raw in zip(
            manifest["nodes"],
            (root_prefix, raw_by_path["root.0"], raw_by_path["root.1"]),
            strict=True,
        ):
            self.assertEqual(
                output["research_notes"][row["raw_start_char"] : row["raw_end_char"]],
                raw,
            )
        for payload in seen_payloads:
            self.assertLessEqual(
                _market_research_web_request_utf8_bytes(
                    system="Исследуй рынок и цитируй источники.",
                    input_json=payload,
                    model_envelope=model_envelope,
                ),
                window_bytes,
            )
        coverage_tamper = copy.deepcopy(output)
        root_row = coverage_tamper["retrieval_manifest"]["nodes"][0]
        root_row["child_task_ids"] = list(reversed(root_row["child_task_ids"]))
        manifest_core = copy.deepcopy(coverage_tamper["retrieval_manifest"])
        manifest_core.pop("manifest_sha256", None)
        coverage_tamper["retrieval_manifest"]["manifest_sha256"] = hashlib.sha256(
            json.dumps(
                manifest_core,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        with self.assertRaisesRegex(OpenRouterError, "lost child coverage"):
            _validate_market_research_web_leaf(
                coverage_tamper,
                source_unit=source_unit,
                requested_site={"domain": "unicode.example"},
                site_profile={"brand_name": "Unicode"},
            )

    def test_market_web_smallest_facet_stops_at_bounded_retry_depth(
        self,
    ) -> None:
        scope = _market_research_web_scope(
            ["customer_jobs"],
            path="root.facet.0",
            facet="discovery_and_problem_definition_jobs",
        )

        children = _market_research_web_children(
            scope,
            predecessor_raw_sha256="a" * 64,
            predecessor_citation_urls=["https://source.example/fact"],
        )
        self.assertEqual([item["refinement_path"] for item in children], ["0", "1"])
        grandchildren = _market_research_web_children(
            children[0],
            predecessor_raw_sha256="b" * 64,
            predecessor_citation_urls=["https://source.example/child"],
        )
        self.assertEqual(grandchildren, [])
        with self.assertRaisesRegex(
            OpenRouterError,
            "segment index is invalid",
        ):
            _market_research_web_scope(
                ["customer_jobs"],
                path="root.facet.0.segment.1",
                facet="discovery_and_problem_definition_jobs",
                segment_index=1,
                predecessor_raw_sha256="a" * 64,
            )

    async def test_market_structuring_shards_preserve_unbounded_tail(
        self,
    ) -> None:
        tail = "MARKET-STRUCTURING-TAIL-7f31"
        long_evidence = ("Подтверждённый рыночный контекст. " * 7_000) + tail
        unit_id = "https://huge.example/:000000"
        structuring_payload = {
            "site_input": {
                "requested_site": {
                    "domain": "huge.example",
                    "url": "https://huge.example/",
                },
                "site_profile": {
                    "brand_name": "Huge",
                    "brand_aliases": ["Huge Platform"],
                    "site_type": "Сайт продукта",
                    "category": "Аналитика",
                    "products": ["Платформа"],
                    "topics": ["AI visibility"],
                    "geography": ["Россия"],
                },
            },
            "evidence_tree": {
                "findings": [
                    {
                        "dimension": "site_confirmed",
                        "claim": "Huge представляет аналитическую платформу.",
                        "evidence": "Huge — аналитическая платформа.",
                        "source_urls": ["https://huge.example/"],
                        "source_unit_ids": [unit_id],
                        "confidence": "high",
                    },
                    {
                        "dimension": "customer_jobs",
                        "claim": "Сравнить поставщиков аналитики.",
                        "evidence": long_evidence,
                        "source_urls": ["https://research.example/market"],
                        "source_unit_ids": [unit_id],
                        "confidence": "medium",
                    },
                ],
                "uncertainties": [],
                "unit_coverage": [
                    {
                        "source_unit_id": unit_id,
                        "state": "evidence",
                        "note": "Фрагмент обработан.",
                    }
                ],
            },
            "evidence_tree_contract": {
                "version": MARKET_RESEARCH_INPUT_HARNESS_VERSION,
                "source_unit_count": 1,
                "source_unit_ids": [unit_id],
            },
            "confirmed_source_urls": ["https://research.example/market"],
            "confirmed_sources": [
                {
                    "url": "https://research.example/market",
                    "title": "Market source",
                    "content": "Подтверждение рынка.",
                }
            ],
        }
        seen_payloads: list[dict[str, Any]] = []

        async def fake_structured_artifact(
            *_args: Any,
            **kwargs: Any,
        ) -> dict[str, Any]:
            shard_payload = copy.deepcopy(kwargs["user_payload"])
            seen_payloads.append(shard_payload)
            return {
                "research": {
                    "status": "limited",
                    "site_confirmed": {
                        "primary_brand": "Huge",
                        "brand_aliases": [],
                        "site_type": "",
                        "category": "",
                        "products": [],
                        "topics": [],
                        "geography": [],
                        "evidence": [],
                    },
                    "external_market_research": {
                        "market": "",
                        "topics": [],
                        "geography": [],
                        "audiences": [],
                        "customer_jobs": [],
                        "decision_criteria": [],
                        "terminology": [],
                        "evidence": [],
                    },
                    "sources": [],
                    "uncertainties": [],
                    "confidence": "low",
                },
                "unit_coverage": [
                    {
                        "source_unit_id": item["source_unit_id"],
                        "core_sha256": item["core_sha256"],
                        "disposition": "structured",
                        "note": "Точный core учтён.",
                    }
                    for item in shard_payload["source_units"]
                ],
            }

        window = {
            "input_utf8_window": 60_000,
            "model_envelope": {
                "context_length": 100_000,
                "max_completion_tokens": 20_000,
            },
            "resolution": "test",
        }
        with (
            patch(
                "app.services.analyzer._final_model_input_window",
                new_callable=AsyncMock,
                return_value=window,
            ),
            patch(
                "app.services.analyzer._structured_artifact",
                new_callable=AsyncMock,
                side_effect=fake_structured_artifact,
            ),
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ),
        ):
            output, plan, raw, usage = await _market_research_structuring_shards(
                "run-id",
                structuring_payload=structuring_payload,
                structuring_system="Структурируй только подтверждённые данные.",
            )

        self.assertGreater(len(seen_payloads), 1)
        self.assertEqual(
            plan["version"],
            MARKET_RESEARCH_STRUCTURING_HARNESS_VERSION,
        )
        self.assertTrue(plan["coverage_complete"])
        self.assertEqual(
            plan["covered_unit_count"],
            plan["source_unit_count"],
        )
        self.assertLessEqual(
            plan["maximum_request_utf8_bytes"],
            window["input_utf8_window"],
        )
        self.assertIsNone(plan["local_corpus_or_shard_count_cap"])
        self.assertIn(
            tail,
            output["external_market_research"]["evidence"][0]["evidence"],
        )
        self.assertIn(tail, raw)
        self.assertIn("_aiv_market_structuring_shards", usage)

    async def test_market_web_cached_receipts_resume_without_rebilling(
        self,
    ) -> None:
        source_unit = {
            "url": "https://resume.example/",
            "title": "Resume",
            "main_text": "Содержательный контекст",
            "_lr_unit_id": "https://resume.example/:000000",
            "_lr_unit_index": 0,
            "_lr_unit_count": 1,
            "_lr_unit_sha256": "c" * 64,
            "_lr_context_sha256": "d" * 64,
            "_lr_start_char": 0,
            "_lr_end_char": 23,
        }
        attestation = {
            "version": WEB_ATTESTATION_VERSION,
            "policy": WebSearchPolicy.REQUIRED.value,
            "state": "verified",
            "metric_eligible": True,
            "web_search_requests": 1,
            "violations": [],
        }
        stored: dict[str, dict[str, Any]] = {}

        async def fake_artifact_output(
            _run_id: str,
            artifact_key: str,
            **_kwargs: Any,
        ) -> Any:
            value = stored.get(artifact_key)
            return copy.deepcopy(value["output_json"]) if value else None

        async def fake_save_artifact(_run_id: str, **kwargs: Any) -> None:
            if (
                kwargs["status"] == "completed"
                and kwargs.get("output_json") is not None
            ):
                stored[kwargs["artifact_key"]] = copy.deepcopy(kwargs)

        async def fake_chat(**kwargs: Any) -> SimpleNamespace:
            payload = json.loads(kwargs["messages"][1]["content"])
            path = payload["research_scope"]["path"]
            result = SimpleNamespace(
                text=f"Фрагмент {path}",
                usage={},
                web_attestation=copy.deepcopy(attestation),
                citations=[{"url": f"https://source.example/{path}"}],
                transport={
                    "output_limited": path == "root",
                    "output_complete": path != "root",
                },
            )
            if path == "root":
                raise OpenRouterOutputLimitError("cut", result=result)
            return result

        kwargs = {
            "source_unit": source_unit,
            "requested_site": {"domain": "resume.example"},
            "site_profile": {"brand_name": "Resume"},
            "system": "Исследуй рынок.",
            "window_bytes": 800_000,
            "model_envelope": {
                "context_length": 1_000_000,
                "max_completion_tokens": 100_000,
            },
        }
        with (
            patch(
                "app.services.analyzer._artifact_output",
                new_callable=AsyncMock,
                side_effect=fake_artifact_output,
            ),
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
                side_effect=fake_save_artifact,
            ),
            patch(
                "app.services.analyzer.chat",
                new_callable=AsyncMock,
                side_effect=fake_chat,
            ) as chat_mock,
        ):
            first = await _market_research_web_leaf("run-id", **kwargs)
            first_calls = chat_mock.await_count
            leaf_key = next(
                key for key in stored if key.startswith("market_research_web_leaf_")
            )
            stored.pop(leaf_key)
            resumed = await _market_research_web_leaf("run-id", **kwargs)

        self.assertEqual(first_calls, 3)
        self.assertEqual(chat_mock.await_count, first_calls)
        self.assertEqual(resumed["research_notes"], first["research_notes"])
        self.assertEqual(resumed["retrieval_manifest"]["node_count"], 3)

    async def test_market_web_cached_coverage_and_raw_tamper_fail_closed(
        self,
    ) -> None:
        source_unit = {
            "url": "https://tamper.example/",
            "title": "Tamper",
            "main_text": "Контекст",
            "_lr_unit_id": "https://tamper.example/:000000",
            "_lr_unit_index": 0,
            "_lr_unit_count": 1,
            "_lr_unit_sha256": "e" * 64,
            "_lr_context_sha256": "f" * 64,
            "_lr_start_char": 0,
            "_lr_end_char": 8,
        }
        attestation = {
            "version": WEB_ATTESTATION_VERSION,
            "policy": WebSearchPolicy.REQUIRED.value,
            "state": "verified",
            "metric_eligible": True,
            "web_search_requests": 1,
            "violations": [],
        }
        stored: dict[str, dict[str, Any]] = {}

        async def load(_run_id: str, key: str, **_kwargs: Any) -> Any:
            return copy.deepcopy(stored[key]["output_json"]) if key in stored else None

        async def save(_run_id: str, **kwargs: Any) -> None:
            if (
                kwargs["status"] == "completed"
                and kwargs.get("output_json") is not None
            ):
                stored[kwargs["artifact_key"]] = copy.deepcopy(kwargs)

        async def complete_chat(**_kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(
                text="Подтверждённый факт",
                usage={},
                web_attestation=copy.deepcopy(attestation),
                citations=[{"url": "https://source.example/fact"}],
                transport={"output_limited": False, "output_complete": True},
            )

        kwargs = {
            "source_unit": source_unit,
            "requested_site": {"domain": "tamper.example"},
            "site_profile": {"brand_name": "Tamper"},
            "system": "Исследуй рынок.",
            "window_bytes": 800_000,
            "model_envelope": {
                "context_length": 1_000_000,
                "max_completion_tokens": 100_000,
            },
        }
        with (
            patch("app.services.analyzer._artifact_output", side_effect=load),
            patch("app.services.analyzer._save_artifact", side_effect=save),
            patch("app.services.analyzer.chat", side_effect=complete_chat) as chat_mock,
        ):
            await _market_research_web_leaf("run-id", **kwargs)
            leaf_key = next(
                key for key in stored if key.startswith("market_research_web_leaf_")
            )
            original_leaf_artifact = copy.deepcopy(stored[leaf_key])
            tampered = stored[leaf_key]["output_json"]
            tampered["research_notes"] += " подмена"
            with self.assertRaisesRegex(OpenRouterError, "notes coverage mismatch"):
                await _market_research_web_leaf("run-id", **kwargs)
            self.assertEqual(chat_mock.await_count, 1)

            stored[leaf_key] = copy.deepcopy(original_leaf_artifact)
            stored[leaf_key]["output_json"]["citations"][0]["title"] = (
                "Подменённый заголовок"
            )
            with self.assertRaisesRegex(OpenRouterError, "citation evidence"):
                await _market_research_web_leaf("run-id", **kwargs)
            self.assertEqual(chat_mock.await_count, 1)

            stored[leaf_key] = copy.deepcopy(original_leaf_artifact)
            stored[leaf_key]["output_json"]["web_attestation"][
                "web_search_requests"
            ] = 999
            with self.assertRaisesRegex(OpenRouterError, "aggregate attestation"):
                await _market_research_web_leaf("run-id", **kwargs)
            self.assertEqual(chat_mock.await_count, 1)

            stored.pop(leaf_key)
            receipt_key = next(
                key for key in stored if key.startswith("market_research_web_attempt_")
            )
            stored[receipt_key]["output_json"]["raw_text"] += " подмена"
            with self.assertRaisesRegex(OpenRouterError, "checksum mismatch"):
                await _market_research_web_leaf("run-id", **kwargs)
            self.assertEqual(chat_mock.await_count, 1)

    async def test_market_web_promotes_durable_post_after_finalize_crash(
        self,
    ) -> None:
        source_unit = {
            "url": "https://checkpoint.example/",
            "title": "Checkpoint",
            "main_text": "Контекст",
            "_lr_unit_id": "https://checkpoint.example/:000000",
            "_lr_unit_index": 0,
            "_lr_unit_count": 1,
            "_lr_unit_sha256": "1" * 64,
            "_lr_context_sha256": "2" * 64,
            "_lr_start_char": 0,
            "_lr_end_char": 8,
        }
        attestation = {
            "version": WEB_ATTESTATION_VERSION,
            "policy": WebSearchPolicy.REQUIRED.value,
            "state": "verified",
            "metric_eligible": True,
            "web_search_requests": 1,
            "violations": [],
        }
        model_envelope = {
            "context_length": 1_000_000,
            "max_completion_tokens": 100_000,
        }
        stored: dict[str, dict[str, Any]] = {}

        async def load(_run_id: str, key: str, **_kwargs: Any) -> Any:
            return copy.deepcopy(stored[key]["output_json"]) if key in stored else None

        async def save(_run_id: str, **kwargs: Any) -> None:
            if (
                kwargs["status"] == "completed"
                and kwargs.get("output_json") is not None
            ):
                stored[kwargs["artifact_key"]] = copy.deepcopy(kwargs)

        async def checkpoint_then_crash(**kwargs: Any) -> SimpleNamespace:
            task_input = json.loads(kwargs["messages"][1]["content"])
            physical_payload = _market_research_web_provider_payload(
                system="Исследуй рынок.",
                input_json=task_input,
                model_envelope=model_envelope,
            )
            citation = {"url": "https://source.example/checkpoint"}
            usage = {
                "_aiv_web_attestation": copy.deepcopy(attestation),
                "_aiv_response_annotations": [
                    {"type": "url_citation", "url_citation": citation}
                ],
            }
            event = {
                "model": ANALYSIS_MODEL,
                "status": "accepted",
                "request_payload": physical_payload,
                "request_sha256": hashlib.sha256(
                    json.dumps(
                        physical_payload,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
                "raw_text": "Уже оплаченный и сохранённый ответ",
                "usage": usage,
                "transport": {
                    "output_limited": False,
                    "output_complete": True,
                },
                "error": None,
            }
            await kwargs["audit_checkpoint"](event)
            raise asyncio.CancelledError

        kwargs = {
            "source_unit": source_unit,
            "requested_site": {"domain": "checkpoint.example"},
            "site_profile": {"brand_name": "Checkpoint"},
            "system": "Исследуй рынок.",
            "window_bytes": 800_000,
            "model_envelope": model_envelope,
        }
        with (
            patch("app.services.analyzer._artifact_output", side_effect=load),
            patch("app.services.analyzer._save_artifact", side_effect=save),
            patch(
                "app.services.analyzer.chat",
                side_effect=checkpoint_then_crash,
            ) as first_chat,
        ):
            with self.assertRaises(asyncio.CancelledError):
                await _market_research_web_leaf("run-id", **kwargs)
            self.assertEqual(first_chat.await_count, 1)

        with (
            patch("app.services.analyzer._artifact_output", side_effect=load),
            patch("app.services.analyzer._save_artifact", side_effect=save),
            patch(
                "app.services.analyzer.chat",
                side_effect=AssertionError("provider must not be called"),
            ) as replay_chat,
        ):
            resumed = await _market_research_web_leaf("run-id", **kwargs)

        replay_chat.assert_not_awaited()
        self.assertEqual(
            resumed["research_notes"],
            "Уже оплаченный и сохранённый ответ",
        )
        self.assertEqual(resumed["retrieval_manifest"]["node_count"], 1)

    def test_prose_evidence_with_embedded_url_counts_as_confirmed(self) -> None:
        # Критик пишет supporting_evidence прозой с URL внутри текста; парсинг
        # строки целиком как URL отвергал даже прошедший ревью набор
        # (прогон 5ae13350, 2026-08-21).
        from app.services.analyzer import _urls_in_evidence_text

        urls = _urls_in_evidence_text(
            "Сайт подтверждает аудиторию, см. https://example.com/news/1. "
            "Второй источник: (https://www.example.org/report)."
        )

        self.assertEqual(
            urls,
            {"https://example.com/news/1", "https://example.org/report"},
        )
        self.assertEqual(_urls_in_evidence_text("просто текст без ссылок"), set())

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

    def test_limited_research_with_verified_attestation_is_usable(self) -> None:
        # limited — штатный честный итог (реальный прогон profi.travel,
        # 2026-08-21): аттестация verified, все измерения покрыты, но часть
        # источников с низкой уверенностью. Фатален только blocked.
        research = _ready_market_research()
        research["status"] = "limited"
        research["sufficiency"]["status"] = "limited"
        research["sufficiency"]["limited_issues"] = [
            "Источник https://example.org/a помечен низкой уверенностью."
        ]

        result = _require_market_research_usable(research)

        self.assertEqual(result["status"], "limited")

    def test_blocked_research_is_never_usable(self) -> None:
        research = _ready_market_research()
        research["status"] = "blocked"
        research["sufficiency"]["status"] = "blocked"
        research["sufficiency"]["blocking_issues"] = [
            "Веб-поиск не прошёл обязательную аттестацию."
        ]

        with self.assertRaises(MarketResearchGateError):
            _require_market_research_usable(research)

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
        profile = _profile_with_offer_contract(profile)
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
        prompt_set = _prompt_set_with_offer_coverage(prompts, profile)
        response = SimpleNamespace(
            parsed=prompt_set,
            text=json.dumps(prompt_set, ensure_ascii=False),
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

        self.assertEqual(result, prompt_set)
        request = chat_mock.await_args.kwargs
        self.assertEqual(request["model"], ANALYSIS_MODEL)
        self.assertEqual(request["reasoning_effort"], "high")
        self.assertFalse(request["retry_response_contract_errors"])
        self.assertFalse(request["retry_transport_errors"])
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
            prompt_set,
            market_research,
        )
        self.assertEqual(
            save_artifact.await_args.kwargs["prompt_version"],
            PROMPT_SET_VERSION,
        )

    async def test_prompt_generation_retries_after_semantic_critic(self) -> None:
        profile = _profile_with_offer_contract(
            {"brand_name": "Example", "brand_aliases": []}
        )
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
            return _prompt_set_with_offer_coverage(prompts, profile)

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
            ) as save_artifact,
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

    async def test_prompt_generator_budget_is_durable_across_restart_faults(
        self,
    ) -> None:
        profile = _profile_with_offer_contract(
            {"brand_name": "Example", "brand_aliases": []}
        )
        research = _ready_market_research()
        artifacts: dict[str, dict[str, Any]] = {}
        events: list[tuple[str, str]] = []

        async def candidate_state(
            _run_id: str,
            *,
            base_payload_digest: str,
        ) -> tuple[int, None, str, dict[str, Any], None, int]:
            matching = [
                value
                for value in artifacts.values()
                if str(value.get("artifact_key") or "").startswith(
                    "prompt_set_candidate_"
                )
                and value.get("input_json", {}).get("base_payload_digest")
                == base_payload_digest
            ]
            attempts = max(
                len(matching),
                max(
                    (
                        int(value["input_json"].get("budget_attempt") or 0)
                        for value in matching
                    ),
                    default=0,
                ),
            )
            return attempts, None, "", {}, None, 0

        async def save_artifact(
            _run_id: str,
            **kwargs: Any,
        ) -> None:
            artifact_key = str(kwargs["artifact_key"])
            artifacts[artifact_key] = copy.deepcopy(kwargs)
            events.append(("artifact", str(kwargs["status"])))

        async def failing_chat(**_kwargs: Any) -> None:
            events.append(("generator", "called"))
            raise RuntimeError("simulated worker crash after reservation")

        recovered = {"recovery": "fable"}
        with (
            patch(
                "app.services.analyzer._artifact_output",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "app.services.analyzer._prompt_candidate_state",
                new=candidate_state,
            ),
            patch(
                "app.services.analyzer._save_artifact",
                new=AsyncMock(side_effect=save_artifact),
            ),
            patch(
                "app.services.analyzer.chat",
                new=AsyncMock(side_effect=failing_chat),
            ) as generator,
            patch(
                "app.services.analyzer._recover_prompt_set",
                new=AsyncMock(return_value=recovered),
            ) as recovery,
            patch(
                "app.services.analyzer._review_prompt_set_semantics",
                new_callable=AsyncMock,
            ) as semantic_review,
        ):
            for _restart in range(4):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "simulated worker crash",
                ):
                    await _generate_prompt_set(
                        "run-id",
                        profile,
                        market_research=research,
                    )
            result = await _generate_prompt_set(
                "run-id",
                profile,
                market_research=research,
            )

        self.assertEqual(result, recovered)
        self.assertEqual(generator.await_count, 4)
        self.assertEqual(len(artifacts), 4)
        self.assertEqual(
            {value["input_json"]["budget_attempt"] for value in artifacts.values()},
            {1, 2, 3, 4},
        )
        self.assertTrue(
            all(value["status"] == "running" for value in artifacts.values())
        )
        for index, event in enumerate(events):
            if event == ("generator", "called"):
                self.assertEqual(events[index - 1], ("artifact", "running"))
        recovery.assert_awaited_once()
        semantic_review.assert_not_awaited()

    async def test_prompt_generator_continues_after_one_shot_contract_failure(
        self,
    ) -> None:
        profile = {
            "brand_name": "Example",
            "brand_aliases": [],
            "category": "Аналитика",
            "customer_jobs": ["Сравнить поставщиков"],
            "decision_criteria": ["Точность"],
            "geography": ["Россия"],
        }
        profile = _profile_with_offer_contract(profile)
        research = _ready_market_research()
        candidate = _deterministic_prompt_fallback(profile, research)
        contract_failure = OpenRouterResponseContractError(
            "Structured response is unusable",
            result=SimpleNamespace(
                text="{unfinished",
                usage={
                    "total_tokens": 17,
                    "_aiv_transport": {"response_id": "response-first"},
                },
            ),
        )
        success = SimpleNamespace(
            parsed=candidate,
            text=json.dumps(candidate, ensure_ascii=False),
            usage={"total_tokens": 23},
        )
        with (
            patch(
                "app.services.analyzer._artifact_output",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "app.services.analyzer._prompt_candidate_state",
                new=AsyncMock(return_value=(0, None, "", {}, None, 0)),
            ),
            patch(
                "app.services.analyzer._reserve_prompt_candidate_attempt",
                new=AsyncMock(side_effect=("candidate-1", "candidate-2")),
            ) as reserve,
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ) as save_artifact,
            patch(
                "app.services.analyzer._save_prompt_candidate",
                new_callable=AsyncMock,
            ),
            patch(
                "app.services.analyzer._save_accepted_prompt_set",
                new_callable=AsyncMock,
            ),
            patch(
                "app.services.analyzer._review_prompt_set_semantics",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "app.services.analyzer.chat",
                new=AsyncMock(side_effect=(contract_failure, success)),
            ) as generator,
        ):
            result = await _generate_prompt_set(
                "run-id",
                profile,
                market_research=research,
            )

        self.assertEqual(result, candidate)
        self.assertEqual(generator.await_count, 2)
        self.assertEqual(reserve.await_count, 2)
        save_artifact.assert_awaited_once()
        failed = save_artifact.await_args.kwargs
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["artifact_key"], "candidate-1")
        self.assertEqual(failed["input_json"]["budget_attempt"], 1)
        self.assertEqual(failed["raw_text"], "{unfinished")
        self.assertEqual(failed["usage_json"]["total_tokens"], 17)
        self.assertEqual(
            failed["usage_json"]["_aiv_transport"]["response_id"],
            "response-first",
        )
        self.assertIn("не вернул пригодный ответ", failed["error_message"])
        for request in generator.await_args_list:
            self.assertFalse(request.kwargs["retry_response_contract_errors"])
            self.assertFalse(request.kwargs["retry_transport_errors"])

    async def test_prompt_generator_exhausts_four_one_shot_failures_before_recovery(
        self,
    ) -> None:
        profile = _profile_with_offer_contract(
            {"brand_name": "Example", "brand_aliases": []}
        )
        research = _ready_market_research()
        recovered = {"recovery": "planned"}
        failures = [
            OpenRouterError("temporary transport failure 1"),
            OpenRouterResponseContractError(
                "incomplete structured response",
                result=SimpleNamespace(text="{", usage={"total_tokens": 3}),
            ),
            ReadTimeout("provider timed out"),
            OpenRouterError("temporary transport failure 4"),
        ]
        with (
            patch(
                "app.services.analyzer._artifact_output",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "app.services.analyzer._prompt_candidate_state",
                new=AsyncMock(return_value=(0, None, "", {}, None, 0)),
            ),
            patch(
                "app.services.analyzer._reserve_prompt_candidate_attempt",
                new=AsyncMock(
                    side_effect=tuple(f"candidate-{attempt}" for attempt in range(1, 5))
                ),
            ) as reserve,
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ) as save_artifact,
            patch(
                "app.services.analyzer._recover_prompt_set",
                new=AsyncMock(return_value=recovered),
            ) as recovery,
            patch(
                "app.services.analyzer._review_prompt_set_semantics",
                new_callable=AsyncMock,
            ) as semantic_review,
            patch(
                "app.services.analyzer.chat",
                new=AsyncMock(side_effect=failures),
            ) as generator,
        ):
            result = await _generate_prompt_set(
                "run-id",
                profile,
                market_research=research,
            )

        self.assertEqual(result, recovered)
        self.assertEqual(generator.await_count, 4)
        self.assertEqual(reserve.await_count, 4)
        self.assertEqual(save_artifact.await_count, 4)
        self.assertTrue(
            all(
                call.kwargs["status"] == "failed"
                for call in save_artifact.await_args_list
            )
        )
        self.assertEqual(
            [
                call.kwargs["input_json"]["budget_attempt"]
                for call in save_artifact.await_args_list
            ],
            [1, 2, 3, 4],
        )
        recovery.assert_awaited_once()
        self.assertIn(
            "попытке 4: OpenRouterError",
            recovery.await_args.kwargs["last_errors"][0],
        )
        semantic_review.assert_not_awaited()

    async def test_recovered_prompt_candidate_reexecutes_semantic_gate(
        self,
    ) -> None:
        profile = _profile_with_offer_contract(
            {"brand_name": "Example", "brand_aliases": []}
        )
        research = _ready_market_research()
        prompts = [
            {
                "prompt_key": f"u-{intent.lower()}",
                "intent_class": intent,
                "role": "unbranded_discovery",
                "text": (
                    f"Какие решения подходят для задачи {intent}? "
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
                "text": f"Что известно об Example для задачи {index}?",
                "rationale": "Проверяет знание бренда.",
                "choice_request": False,
            }
            for index, intent in enumerate(("I", "E", "TR"), start=1)
        )
        candidate = _prompt_set_with_offer_coverage(prompts, profile)
        artifacts: dict[str, dict[str, Any]] = {}
        events: list[str] = []

        async def candidate_state(
            _run_id: str,
            *,
            base_payload_digest: str,
        ) -> tuple[
            int,
            dict[str, Any] | None,
            str,
            dict[str, Any],
            str | None,
            int,
        ]:
            matching = [
                value
                for value in artifacts.values()
                if str(value.get("artifact_key") or "").startswith(
                    "prompt_set_candidate_"
                )
                and value.get("input_json", {}).get("base_payload_digest")
                == base_payload_digest
            ]
            if not matching:
                return 0, None, "", {}, None, 0
            latest = max(
                matching,
                key=lambda value: int(value["input_json"].get("budget_attempt") or 0),
            )
            budget_attempt = int(latest["input_json"].get("budget_attempt") or 0)
            return (
                len(matching),
                copy.deepcopy(latest.get("output_json")),
                str(latest.get("raw_text") or ""),
                copy.deepcopy(latest.get("usage_json") or {}),
                str(latest["artifact_key"]),
                budget_attempt,
            )

        async def save_artifact(
            _run_id: str,
            **kwargs: Any,
        ) -> None:
            artifacts[str(kwargs["artifact_key"])] = copy.deepcopy(kwargs)
            metadata = (kwargs.get("usage_json") or {}).get("_aiv_prompt_candidate")
            if isinstance(metadata, dict):
                events.append(str(metadata.get("reservation_state") or ""))

        async def semantic_review(
            _run_id: str,
            _profile: dict[str, Any],
            _candidate: dict[str, Any],
            _research: dict[str, Any],
        ) -> list[str]:
            events.append("semantic_gate")
            if events.count("semantic_gate") == 1:
                raise RuntimeError("simulated crash inside semantic gate")
            return []

        original_validate = _validate_prompt_set

        def structural_review(
            value: dict[str, Any],
            value_profile: dict[str, Any],
        ) -> list[str]:
            events.append("structural_gate")
            return original_validate(value, value_profile)

        response = SimpleNamespace(
            parsed=candidate,
            text=json.dumps(candidate, ensure_ascii=False),
            usage={"total_tokens": 1},
        )
        with (
            patch(
                "app.services.analyzer._artifact_output",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "app.services.analyzer._prompt_candidate_state",
                new=candidate_state,
            ),
            patch(
                "app.services.analyzer._save_artifact",
                new=AsyncMock(side_effect=save_artifact),
            ),
            patch(
                "app.services.analyzer.chat",
                new_callable=AsyncMock,
                return_value=response,
            ) as generator,
            patch(
                "app.services.analyzer._validate_prompt_set",
                new=structural_review,
            ),
            patch(
                "app.services.analyzer._review_prompt_set_semantics",
                new=AsyncMock(side_effect=semantic_review),
            ) as semantic_gate,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "simulated crash inside semantic gate",
            ):
                await _generate_prompt_set(
                    "run-id",
                    profile,
                    market_research=research,
                )
            result = await _generate_prompt_set(
                "run-id",
                profile,
                market_research=research,
            )

        self.assertEqual(result, candidate)
        self.assertEqual(generator.await_count, 1)
        self.assertEqual(semantic_gate.await_count, 2)
        self.assertEqual(events.count("structural_gate"), 2)
        self.assertLess(
            events.index("candidate_saved"),
            events.index("semantic_gate"),
        )
        accepted = artifacts["prompt_set"]
        accepted_metadata = accepted["usage_json"]["_aiv_prompt_candidate"]
        self.assertTrue(accepted_metadata["recovered_after_worker_restart"])
        self.assertTrue(accepted_metadata["structural_gate_reexecuted"])
        self.assertTrue(accepted_metadata["semantic_gate_reexecuted"])

    def test_deterministic_prompt_fallback_is_valid_and_unbranded(self) -> None:
        profile = {
            "brand_name": "Example",
            "brand_aliases": ["Example Lab"],
            "category": "Аналитика Example",
            "customer_jobs": ["Сравнить поставщиков аналитики"],
            "decision_criteria": ["Точность"],
            "geography": ["Россия"],
        }
        profile = _profile_with_offer_contract(profile)

        prompt_set = _deterministic_prompt_fallback(
            profile,
            _ready_market_research(),
        )

        self.assertEqual(_validate_prompt_set(prompt_set, profile), [])
        unbranded = [
            item
            for item in prompt_set["prompts"]
            if item["role"] == "unbranded_discovery"
        ]
        self.assertEqual(len(unbranded), 6)
        self.assertEqual(
            {item["intent_class"] for item in unbranded},
            {"I", "E", "T", "NB", "NAV", "TR"},
        )
        self.assertTrue(all(item["choice_request"] for item in unbranded))
        self.assertTrue(
            all("example" not in item["text"].casefold() for item in unbranded)
        )

    async def test_deterministic_prompt_fallback_has_separate_cache_provenance(
        self,
    ) -> None:
        profile = {
            "brand_name": "Example",
            "brand_aliases": [],
            "category": "Аналитика",
            "customer_jobs": ["Сравнить поставщиков"],
            "decision_criteria": ["Точность"],
            "geography": ["Россия"],
        }
        profile = _profile_with_offer_contract(profile)
        research = _ready_market_research()
        fallback = _deterministic_prompt_fallback(profile, research)
        queried_models: list[str] = []

        async def fake_artifact_output(
            _run_id: str,
            _artifact_key: str,
            **kwargs: object,
        ) -> dict[str, object] | None:
            model = str(kwargs.get("model") or "")
            queried_models.append(model)
            return fallback if model == "deterministic/prompt-fallback-v1" else None

        with (
            patch(
                "app.services.analyzer._artifact_output",
                new=fake_artifact_output,
            ),
            patch(
                "app.services.analyzer._review_prompt_set_semantics",
                new_callable=AsyncMock,
                return_value=[],
            ) as semantic_review,
            patch(
                "app.services.analyzer.chat",
                new_callable=AsyncMock,
            ) as chat_mock,
        ):
            result = await _generate_prompt_set(
                "run-id",
                profile,
                market_research=research,
            )

        self.assertEqual(result, fallback)
        self.assertEqual(
            queried_models,
            [ANALYSIS_MODEL, "deterministic/prompt-fallback-v1"],
        )
        semantic_review.assert_awaited_once()
        chat_mock.assert_not_awaited()

    async def test_prompt_recovery_executes_only_the_planner_allowlist(
        self,
    ) -> None:
        profile = {
            "brand_name": "Example",
            "brand_aliases": [],
            "category": "Аналитика",
            "customer_jobs": ["Сравнить поставщиков"],
            "decision_criteria": ["Точность"],
            "geography": ["Россия"],
        }
        profile = _profile_with_offer_contract(profile)
        research = _ready_market_research()
        payload = {
            "requested_site": {},
            "site_profile": profile,
            "market_research": research,
            "market_research_digest": "a" * 64,
        }
        plan = SimpleNamespace(
            epoch=1,
            decision={
                "action": ACTION_DETERMINISTIC_FALLBACK,
                "rationale": "Локальный цикл не сошёлся.",
                "guidance": "",
                "acceptance_checks": [
                    "prompt_contract_valid",
                    "semantic_review_passed",
                ],
            },
        )
        with (
            patch(
                "app.services.analyzer.plan_durable_recovery",
                new=AsyncMock(return_value=plan),
            ) as planner,
            patch(
                "app.services.analyzer.mark_recovery_executing",
                new_callable=AsyncMock,
            ) as mark,
            patch(
                "app.services.analyzer.finish_recovery",
                new_callable=AsyncMock,
            ) as finish,
            patch(
                "app.services.analyzer._save_accepted_prompt_set",
                new_callable=AsyncMock,
            ) as save,
            patch(
                "app.services.analyzer.chat",
                new_callable=AsyncMock,
            ) as chat_mock,
            patch(
                "app.services.analyzer._review_prompt_set_semantics",
                new_callable=AsyncMock,
                return_value=[],
            ) as semantic_review,
        ):
            recovered = await _recover_prompt_set(
                "run-id",
                profile=profile,
                research=research,
                payload=payload,
                system="Системный контракт",
                previous_set=None,
                last_errors=["Критик отклонил набор."],
            )

        planner.assert_awaited_once()
        self.assertEqual(
            planner.await_args.kwargs["allowed_actions"],
            {
                ACTION_RETRY_WITH_GUIDANCE,
                ACTION_DETERMINISTIC_FALLBACK,
                ACTION_STOP,
            },
        )
        self.assertEqual(
            planner.await_args.kwargs["stage_planner_call_limit"],
            1,
        )
        mark.assert_awaited_once_with(plan)
        finish.assert_awaited_once()
        self.assertTrue(finish.await_args.kwargs["succeeded"])
        save.assert_awaited_once()
        self.assertEqual(
            save.await_args.kwargs["model"],
            "deterministic/prompt-fallback-v1",
        )
        semantic_review.assert_awaited_once_with(
            "run-id",
            profile,
            recovered,
            research,
        )
        chat_mock.assert_not_awaited()
        self.assertEqual(_validate_prompt_set(recovered, profile), [])

    async def test_prompt_recovery_uses_verified_code_owned_fallback_when_fable_fails(
        self,
    ) -> None:
        profile = _profile_with_offer_contract(
            {
                "brand_name": "Example",
                "brand_aliases": [],
                "category": "Аналитика",
                "customer_jobs": ["Сравнить поставщиков"],
                "decision_criteria": ["Точность"],
                "geography": ["Россия"],
            }
        )
        research = _ready_market_research()
        payload = {
            "requested_site": {},
            "site_profile": profile,
            "market_research": research,
            "market_research_digest": "a" * 64,
        }
        plan = SimpleNamespace(
            epoch=2,
            decision={
                "action": ACTION_DETERMINISTIC_FALLBACK,
                "rationale": "Кодовый fallback сохраняет строгие проверки.",
                "guidance": "",
                "acceptance_checks": [
                    "prompt_contract_valid",
                    "semantic_review_passed",
                ],
            },
        )
        with (
            patch(
                "app.services.analyzer.plan_durable_recovery",
                new=AsyncMock(
                    side_effect=RecoveryPlannerUnavailable("provider unavailable")
                ),
            ) as planner,
            patch(
                "app.services.analyzer.plan_code_owned_recovery",
                new=AsyncMock(return_value=plan),
            ) as fallback,
            patch(
                "app.services.analyzer.mark_recovery_executing",
                new_callable=AsyncMock,
            ),
            patch(
                "app.services.analyzer.finish_recovery",
                new_callable=AsyncMock,
            ),
            patch(
                "app.services.analyzer._save_accepted_prompt_set",
                new_callable=AsyncMock,
            ),
            patch(
                "app.services.analyzer._review_prompt_set_semantics",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            recovered = await _recover_prompt_set(
                "run-id",
                profile=profile,
                research=research,
                payload=payload,
                system="Системный контракт",
                previous_set=None,
                last_errors=["Критик отклонил набор."],
            )

        planner.assert_awaited_once()
        fallback.assert_awaited_once()
        fallback_kwargs = fallback.await_args.kwargs
        self.assertEqual(
            fallback_kwargs["decision"]["action"],
            ACTION_DETERMINISTIC_FALLBACK,
        )
        self.assertEqual(
            set(fallback_kwargs["decision"]["acceptance_checks"]),
            {"prompt_contract_valid", "semantic_review_passed"},
        )
        self.assertEqual(_validate_prompt_set(recovered, profile), [])

    async def test_guided_prompt_recovery_disables_internal_chat_retries(
        self,
    ) -> None:
        profile = {
            "brand_name": "Example",
            "brand_aliases": [],
            "category": "Аналитика",
            "customer_jobs": ["Сравнить поставщиков"],
            "decision_criteria": ["Точность"],
            "geography": ["Россия"],
        }
        profile = _profile_with_offer_contract(profile)
        research = _ready_market_research()
        payload = {
            "requested_site": {},
            "site_profile": profile,
            "market_research": research,
            "market_research_digest": "a" * 64,
        }
        recovered = _deterministic_prompt_fallback(profile, research)
        result = SimpleNamespace(
            parsed=recovered,
            text=json.dumps(recovered, ensure_ascii=False),
            usage={"total_tokens": 1},
        )
        plan = SimpleNamespace(
            epoch=1,
            decision={
                "action": ACTION_RETRY_WITH_GUIDANCE,
                "rationale": "Нужна последняя точечная правка.",
                "guidance": "Сохранить валидные сценарии дословно.",
                "acceptance_checks": [
                    "prompt_contract_valid",
                    "semantic_review_passed",
                ],
            },
        )
        with (
            patch(
                "app.services.analyzer.plan_durable_recovery",
                new=AsyncMock(return_value=plan),
            ),
            patch(
                "app.services.analyzer.mark_recovery_executing",
                new_callable=AsyncMock,
            ),
            patch(
                "app.services.analyzer.finish_recovery",
                new_callable=AsyncMock,
            ),
            patch(
                "app.services.analyzer._save_accepted_prompt_set",
                new_callable=AsyncMock,
            ),
            patch(
                "app.services.analyzer._review_prompt_set_semantics",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "app.services.analyzer.chat",
                new=AsyncMock(return_value=result),
            ) as chat_mock,
        ):
            actual = await _recover_prompt_set(
                "run-id",
                profile=profile,
                research=research,
                payload=payload,
                system="Системный контракт",
                previous_set=None,
                last_errors=["Критик отклонил набор."],
            )

        self.assertEqual(actual, recovered)
        chat_mock.assert_awaited_once()
        request = chat_mock.await_args.kwargs
        self.assertFalse(request["retry_response_contract_errors"])
        self.assertFalse(request["retry_transport_errors"])

    async def test_prompt_recovery_rejects_unexecuted_acceptance_checks(
        self,
    ) -> None:
        profile = {
            "brand_name": "Example",
            "brand_aliases": [],
            "category": "Аналитика",
        }
        profile = _profile_with_offer_contract(profile)
        research = _ready_market_research()
        payload = {
            "requested_site": {},
            "site_profile": profile,
            "market_research": research,
            "market_research_digest": "a" * 64,
        }
        plan = SimpleNamespace(
            epoch=1,
            decision={
                "action": ACTION_DETERMINISTIC_FALLBACK,
                "rationale": "Локальный цикл не сошёлся.",
                "guidance": "",
                "acceptance_checks": ["prompt_contract_valid"],
            },
        )
        with (
            patch(
                "app.services.analyzer.plan_durable_recovery",
                new=AsyncMock(return_value=plan),
            ),
            patch(
                "app.services.analyzer.mark_recovery_executing",
                new_callable=AsyncMock,
            ),
            patch(
                "app.services.analyzer.finish_recovery",
                new_callable=AsyncMock,
            ) as finish,
            patch(
                "app.services.analyzer._save_accepted_prompt_set",
                new_callable=AsyncMock,
            ) as save,
            self.assertRaises(OpenRouterError) as raised,
        ):
            await _recover_prompt_set(
                "run-id",
                profile=profile,
                research=research,
                payload=payload,
                system="Системный контракт",
                previous_set=None,
                last_errors=["Критик отклонил набор."],
            )

        self.assertIn("semantic_review_passed", str(raised.exception))
        self.assertFalse(finish.await_args.kwargs["succeeded"])
        save.assert_not_awaited()

    async def test_prompt_recovery_rejects_extra_acceptance_checks(
        self,
    ) -> None:
        profile = {
            "brand_name": "Example",
            "brand_aliases": [],
            "category": "Аналитика",
        }
        profile = _profile_with_offer_contract(profile)
        research = _ready_market_research()
        payload = {
            "requested_site": {},
            "site_profile": profile,
            "market_research": research,
            "market_research_digest": "a" * 64,
        }
        plan = SimpleNamespace(
            epoch=1,
            decision={
                "action": ACTION_DETERMINISTIC_FALLBACK,
                "rationale": "Локальный цикл не сошёлся.",
                "guidance": "",
                "acceptance_checks": [
                    "prompt_contract_valid",
                    "semantic_review_passed",
                    "critic_gate_passed",
                ],
            },
        )
        with (
            patch(
                "app.services.analyzer.plan_durable_recovery",
                new=AsyncMock(return_value=plan),
            ),
            patch(
                "app.services.analyzer.mark_recovery_executing",
                new_callable=AsyncMock,
            ),
            patch(
                "app.services.analyzer.finish_recovery",
                new_callable=AsyncMock,
            ) as finish,
            patch(
                "app.services.analyzer._save_accepted_prompt_set",
                new_callable=AsyncMock,
            ) as save,
            self.assertRaises(OpenRouterError) as raised,
        ):
            await _recover_prompt_set(
                "run-id",
                profile=profile,
                research=research,
                payload=payload,
                system="Системный контракт",
                previous_set=None,
                last_errors=["Критик отклонил набор."],
            )

        self.assertIn("unsupported: critic_gate_passed", str(raised.exception))
        self.assertFalse(finish.await_args.kwargs["succeeded"])
        self.assertEqual(
            finish.await_args.kwargs["details"]["unsupported_acceptance_checks"],
            ["critic_gate_passed"],
        )
        save.assert_not_awaited()

    async def test_market_research_digest_invalidates_cached_prompt_set(
        self,
    ) -> None:
        profile = _profile_with_offer_contract(
            {"brand_name": "Example", "brand_aliases": []}
        )

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
            return _prompt_set_with_offer_coverage(prompts, profile)

        cached = prompt_set("из кэша")
        regenerated = prompt_set("после изменения")
        research_before = _ready_market_research(
            first_job="Сравнить поставщиков аналитики"
        )
        research_after = json.loads(json.dumps(research_before, ensure_ascii=False))
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
        calls: list[str] = []

        async def fake_processing(
            _run_id: str,
            **kwargs: Any,
        ) -> dict[str, Any]:
            calls.append(str(kwargs["artifact_key"]))
            payload = kwargs["user_payload"]
            if "answers" in payload:
                claim = payload["answers"][0]["core_claim"]
                return {
                    "catalog": {
                        "target_aliases": ["Example"],
                        "entities": [],
                        "uncertainties": ["Example"],
                    },
                    "core_dispositions": [
                        {
                            "claim_id": claim["claim_id"],
                            "unit_id": claim["unit_id"],
                            "core_sha256": claim["core_sha256"],
                            "disposition": "grounded_fact",
                            "evidence_quote": "Example",
                            "reason": "В core буквально назван бренд Example.",
                        }
                    ],
                }
            return {
                "target_aliases": ["Example"],
                "entities": [],
                "uncertainties": [],
            }

        with patch(
            "app.services.analyzer._processing_artifact",
            new=fake_processing,
        ):
            await _entity_catalog(
                "run-id",
                {"brand_name": "Example", "brand_aliases": [], "products": []},
                [{"answer_id": 1, "answer": "Example"}],
            )

        self.assertEqual(len(calls), 2)

    async def test_entity_catalog_repairs_unique_markdown_only_quote(self) -> None:
        calls: list[tuple[str, str]] = []
        leaf_catalog = {
            "target_aliases": [],
            "entities": [
                {
                    "canonical_name": "ST Tattoo",
                    "aliases": [],
                    "category": "competitor",
                    "target_relationship": "competitor",
                    "commercially_relevant": True,
                    "mention_policy": "standalone",
                    "evidence": "«ST Tattoo, Tattoo Roko и другие»",
                },
                {
                    "canonical_name": "Tattoo Roko",
                    "aliases": [],
                    "category": "competitor",
                    "target_relationship": "competitor",
                    "commercially_relevant": True,
                    "mention_policy": "standalone",
                    "evidence": "«ST Tattoo, Tattoo Roko и другие»",
                },
            ],
            "uncertainties": [],
        }

        async def fake_processing(
            _run_id: str,
            **kwargs: Any,
        ) -> dict[str, Any]:
            calls.append((str(kwargs["artifact_key"]), str(kwargs["prompt_version"])))
            payload = kwargs["user_payload"]
            if "answers" in payload:
                claim = payload["answers"][0]["core_claim"]
                self.assertEqual(claim["unit_id"], "614:000000")
                return {
                    "catalog": copy.deepcopy(leaf_catalog),
                    "core_dispositions": [
                        {
                            "claim_id": claim["claim_id"],
                            "unit_id": claim["unit_id"],
                            "core_sha256": claim["core_sha256"],
                            "disposition": "grounded_fact",
                            "evidence_quote": "ST Tattoo, Tattoo Roko",
                            "reason": ("В core перечислены две альтернативы."),
                        }
                    ],
                }
            return copy.deepcopy(leaf_catalog)

        with patch(
            "app.services.analyzer._processing_artifact",
            new=fake_processing,
        ):
            result = await _entity_catalog(
                "run-id",
                {
                    "brand_name": "Makar's Tattoo",
                    "brand_aliases": [],
                    "products": [],
                },
                [
                    {
                        "answer_id": 614,
                        "answer": (
                            "Среди альтернатив названы *ST Tattoo*, "
                            "*Tattoo Roko* и другие."
                        ),
                    }
                ],
            )

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][1], ENTITY_CATALOG_CHUNK_VERSION)
        self.assertIn(
            "*ST Tattoo*, *Tattoo Roko*",
            "\n".join(result["uncertainties"]),
        )

    async def test_entity_catalog_packs_by_full_request_bytes_before_merge(
        self,
    ) -> None:
        answers = [
            {
                "answer_id": index + 1,
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
                return _deterministic_entity_catalog_union(
                    list(
                        kwargs["user_payload"][  # type: ignore[index]
                            "chunk_catalogs"
                        ]
                    )
                )

            chunk = kwargs["user_payload"]["answers"]  # type: ignore[index]
            self.assertLessEqual(1_000 + len(chunk) * 1_000, 9_000)
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
                    "catalog": {
                        "target_aliases": [str(chunk[0]["core_claim"]["core_text"])],
                        "entities": [],
                        "uncertainties": [
                            str(item["core_claim"]["core_text"]) for item in chunk
                        ],
                    },
                    "core_dispositions": [
                        {
                            "claim_id": item["core_claim"]["claim_id"],
                            "unit_id": item["core_claim"]["unit_id"],
                            "core_sha256": item["core_claim"]["core_sha256"],
                            "disposition": "grounded_fact",
                            "evidence_quote": item["core_claim"]["core_text"],
                            "reason": (
                                "В core буквально присутствует именованная "
                                "сущность для каталога."
                            ),
                        }
                        for item in chunk
                    ],
                }
            finally:
                active -= 1

        with (
            patch(
                "app.services.analyzer._processing_artifact",
                new=fake_processing,
            ),
            patch(
                "app.services.analyzer._analyzer_model_input_window",
                new=AsyncMock(
                    return_value={
                        "input_utf8_window": 9_000,
                        "model_envelope": {
                            "context_length": 12_000,
                            "max_completion_tokens": 3_000,
                        },
                    }
                ),
            ),
            patch(
                "app.services.analyzer._structured_provider_request_utf8_bytes",
                side_effect=lambda **kwargs: (
                    1_000 + len(kwargs["user_payload"].get("answers", [])) * 1_000
                ),
            ),
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

        self.assertEqual(
            result["target_aliases"],
            ["Named entity 0", "Named entity 8", "Named entity 16"],
        )
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
            ["Named entity 0", "Named entity 8", "Named entity 16"],
        )

    async def test_site_profile_reducer_uses_byte_tree_and_lossless_terminal_union(
        self,
    ) -> None:
        pages = [
            {
                "url": f"https://example.com/page-{index}",
                "main_text": f"Page {index} " + ("raw " * 1_000),
            }
            for index in range(30)
        ]
        reducer_payloads: list[tuple[str, dict[str, Any]]] = []

        def leaf_profile(name: str) -> dict[str, Any]:
            return {
                "brand_name": "Example",
                "brand_aliases": [],
                "site_type": "service",
                "category": "test",
                "topics": ["topic"],
                "market": "market",
                "business_model": "b2b",
                "products": [f"product:{name}"],
                "audiences": [],
                "customer_jobs": [],
                "decision_criteria": [],
                "geography": [],
                "language": "ru",
                "positioning": "positioning",
                "entity_scope": [],
                "evidence": [name + ":" + ("e" * 2_200)],
                "uncertainties": [],
                "confidence": "medium",
            }

        async def fake_structured(
            run_id: str,
            **kwargs: Any,
        ) -> dict[str, Any]:
            self.assertEqual(run_id, "run-id")
            payload = kwargs["user_payload"]
            if "page_unit" in payload:
                claim = payload["core_claim"]
                profile = leaf_profile(str(payload["page_unit"]["url"]))
                quote = str(claim["core_text"]).split(" ", 2)[0]
                profile["evidence"].append(quote)
                return {
                    "profile": profile,
                    "core_disposition": {
                        "claim_id": claim["claim_id"],
                        "unit_id": claim["unit_id"],
                        "core_sha256": claim["core_sha256"],
                        "disposition": "grounded_fact",
                        "evidence_quote": quote,
                        "reason": (
                            "В core буквально присутствует профильный факт страницы."
                        ),
                    },
                }
            reducer_payloads.append((str(kwargs["system"]), copy.deepcopy(payload)))
            return _deterministic_site_profile_union(list(payload["partial_profiles"]))

        envelope = {
            "context_length": 40_000,
            "max_completion_tokens": 2_000,
        }
        with (
            patch(
                "app.services.analyzer._structured_artifact",
                new=fake_structured,
            ),
            patch(
                "app.services.analyzer.model_output_envelope",
                new=AsyncMock(return_value=envelope),
            ),
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ) as save_artifact,
        ):
            result = await _classify_site(
                "run-id",
                {
                    "requested_site": {
                        "domain": "example.com",
                        "url": "https://example.com/",
                    },
                    "pages": pages,
                },
            )

        self.assertEqual(len(result["products"]), len(pages))
        self.assertTrue(reducer_payloads)
        safe_window = 40_000 - 2_000
        for system, payload in reducer_payloads:
            self.assertLessEqual(
                _serialized_llm_request_bytes(
                    system=system,
                    user_payload=payload,
                ),
                safe_window,
            )
        # The model sees compact digests; complete manifests and ids remain
        # code-owned even when the terminal union is required.
        for _system, payload in reducer_payloads:
            reduction = payload["reduction"]
            self.assertEqual(
                reduction["version"],
                ANALYZER_REDUCER_HARNESS_VERSION,
            )
            self.assertNotIn("source_unit_ids", reduction)
            self.assertNotIn("source_manifests", reduction)
        self.assertEqual(
            save_artifact.await_args.kwargs["usage_json"]["_aiv_analyzer_reducer"][
                "mode"
            ],
            "deterministic_terminal_union",
        )

    async def test_entity_catalog_reducer_uses_byte_tree_without_losing_leaves(
        self,
    ) -> None:
        answers = [
            {"answer_id": index + 1, "answer": f"Competitor {index}"}
            for index in range(80)
        ]
        reducer_requests: list[tuple[str, dict[str, Any]]] = []

        async def fake_processing(
            run_id: str,
            **kwargs: Any,
        ) -> dict[str, Any]:
            self.assertEqual(run_id, "run-id")
            payload = kwargs["user_payload"]
            if "answers" in payload:
                return {
                    "catalog": {
                        "target_aliases": [],
                        "entities": [
                            {
                                "canonical_name": (
                                    str(item["core_claim"]["core_text"])
                                ),
                                "aliases": [],
                                "category": "competitor",
                                "target_relationship": "competitor",
                                "commercially_relevant": True,
                                "mention_policy": "standalone",
                                "evidence": (
                                    "«"
                                    + str(item["core_claim"]["core_text"])
                                    + "»"
                                    + " — "
                                    + ("e" * 2_000)
                                ),
                            }
                            for item in payload["answers"]
                        ],
                        "uncertainties": [],
                    },
                    "core_dispositions": [
                        {
                            "claim_id": item["core_claim"]["claim_id"],
                            "unit_id": item["core_claim"]["unit_id"],
                            "core_sha256": item["core_claim"]["core_sha256"],
                            "disposition": "grounded_fact",
                            "evidence_quote": item["core_claim"]["core_text"],
                            "reason": (
                                "В core буквально присутствует имя "
                                "конкурента для каталога."
                            ),
                        }
                        for item in payload["answers"]
                    ],
                }
            reducer_requests.append((str(kwargs["system"]), copy.deepcopy(payload)))
            return _deterministic_entity_catalog_union(list(payload["chunk_catalogs"]))

        envelope = {
            "context_length": 40_000,
            "max_completion_tokens": 2_000,
        }
        with (
            patch(
                "app.services.analyzer._processing_artifact",
                new=fake_processing,
            ),
            patch(
                "app.services.analyzer.model_output_envelope",
                new=AsyncMock(return_value=envelope),
            ),
            patch(
                "app.services.analyzer._structured_provider_request_utf8_bytes",
                side_effect=lambda **kwargs: (
                    20_000 + len(kwargs["user_payload"].get("answers", [])) * 6_000
                    if "answers" in kwargs["user_payload"]
                    else _structured_provider_request_utf8_bytes(**kwargs)
                ),
            ),
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ) as save_artifact,
        ):
            result = await _entity_catalog(
                "run-id",
                {
                    "brand_name": "Example",
                    "brand_aliases": [],
                    "products": [],
                    "topics": [],
                    "entity_scope": [],
                },
                answers,
            )

        self.assertEqual(len(result["entities"]), len(answers))
        self.assertTrue(reducer_requests)
        safe_window = 40_000 - 2_000
        for system, payload in reducer_requests:
            self.assertLessEqual(
                _serialized_llm_request_bytes(
                    system=system,
                    user_payload=payload,
                ),
                safe_window,
            )
            self.assertNotIn("source_unit_ids", payload["reduction"])
            self.assertNotIn("partition_manifests", payload["reduction"])
        self.assertEqual(
            save_artifact.await_args.kwargs["usage_json"]["_aiv_analyzer_reducer"][
                "mode"
            ],
            "deterministic_terminal_union",
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

        envelope = {
            "context_length": 80_000,
            "max_completion_tokens": 16_000,
        }
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
            patch(
                "app.services.analyzer.model_output_envelope",
                new=AsyncMock(return_value=envelope),
            ),
            patch(
                "app.services.analyzer._structured_provider_request_utf8_bytes",
                side_effect=lambda **kwargs: (
                    1_000 + len(kwargs["user_payload"].get("answers", [])) * 5_000
                ),
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
        # Partition requests may complete independently, but persistence is
        # one code-owned all-or-nothing write for the reconstructed answers.
        self.assertEqual(save_order, [1])
        progress_values = [call.kwargs["percent"] for call in progress.await_args_list]
        self.assertEqual(progress_values[0], 76)
        self.assertEqual(progress_values[-1], 80)
        self.assertEqual(progress_values, sorted(progress_values))

    async def test_annotation_preflights_full_wrapper_and_hash_joins_context_shards(
        self,
    ) -> None:
        profile = {
            "brand_name": "Example",
            "brand_aliases": ["Example brand"],
            "site_type": "service",
            "category": "test",
            "topics": ["analytics"],
            "market": "test market",
            "business_model": "b2b",
            "products": [],
            "audiences": [],
            "customer_jobs": [],
            "decision_criteria": [],
            "geography": [],
            "language": "ru",
            "positioning": "test",
            "entity_scope": [],
            "evidence": [],
            "uncertainties": [],
            "confidence": "medium",
        }
        catalog = {
            "target_aliases": ["Example"],
            "entities": [
                {
                    "canonical_name": f"Competitor {index}",
                    "aliases": [f"C{index}"],
                    "category": "competitor",
                    "target_relationship": "competitor",
                    "commercially_relevant": True,
                    "mention_policy": "standalone",
                    "evidence": "e" * 900,
                }
                for index in range(80)
            ],
            "uncertainties": [],
        }
        raw_answer = "Example подробно описан в ответе. Упомянут Competitor 79."
        envelope = {
            "context_length": 80_000,
            "max_completion_tokens": 16_000,
        }
        pending = [
            {
                "answer_id": 1,
                "mode": "memory",
                "system": "Test",
                "answer_model": "test/model",
                "answer_sha256": hashlib.sha256(raw_answer.encode("utf-8")).hexdigest(),
                "scenario": "Что известно про Example?",
                "scenario_role": "brand_diagnostic",
                "intent_class": "I",
                "answer": raw_answer,
            }
        ]
        request_payloads: list[tuple[str, dict[str, Any]]] = []
        saved_annotations: list[dict[str, Any]] = []

        async def fake_processing(
            run_id: str,
            **kwargs: Any,
        ) -> dict[str, Any]:
            self.assertEqual(run_id, "run-id")
            payload = copy.deepcopy(kwargs["user_payload"])
            request_payloads.append((str(kwargs["system"]), payload))
            return {
                "answers": [
                    {
                        "answer_id": item["answer_id"],
                        "valid": True,
                        "target_mentioned": True,
                        "target_position": None,
                        "target_role": "mentioned",
                        "sentiment": "neutral",
                        "entity_mentions": [],
                        "brand_answer": {
                            "directness": "direct",
                            "specificity": "generic",
                            "supported_facets": [],
                            "contradictions": [],
                        },
                        "evidence": ["Example"],
                        "uncertainties": [],
                    }
                    for item in payload["answers"]
                ]
            }

        async def fake_save(
            run_id: str,
            annotations: list[dict[str, Any]],
            allowed_ids: set[int],
        ) -> int:
            self.assertEqual(run_id, "run-id")
            self.assertEqual(allowed_ids, {1})
            saved_annotations.extend(copy.deepcopy(annotations))
            return len(annotations)

        class FakeResult:
            def all(self) -> list[tuple[SimpleNamespace, SimpleNamespace]]:
                return [
                    (
                        SimpleNamespace(
                            response_text=raw_answer,
                            model="test/model",
                        ),
                        SimpleNamespace(annotation_json=saved_annotations[0]),
                    )
                ]

        class FakeSession:
            async def execute(self, statement: object) -> FakeResult:
                del statement
                return FakeResult()

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
            ),
            patch(
                "app.services.analyzer.SessionLocal",
                new=lambda: FakeSessionContext(),
            ),
            patch(
                "app.services.analyzer.model_output_envelope",
                new=AsyncMock(return_value=envelope),
            ),
        ):
            await _annotate_answers("run-id", profile, catalog)

        # The 79 irrelevant catalog rows are not repeated beside this answer;
        # the literal tail entity and identity profile remain available.
        self.assertLessEqual(len(request_payloads), 2)
        seen_names: set[str] = set()
        request_ids: list[int] = []
        manifest_digests: set[str] = set()
        for system, payload in request_payloads:
            self.assertLessEqual(
                _structured_provider_request_utf8_bytes(
                    model=PROCESSING_MODEL,
                    model_envelope=envelope,
                    system=system,
                    user_payload=payload,
                    schema=ANNOTATION_SCHEMA,
                    schema_name="aiv_annotations_0000000000000000",
                    reasoning_effort="high",
                    temperature=0.15,
                ),
                64_000,
            )
            self.assertEqual(payload["answers"][0]["answer"], raw_answer)
            request_ids.append(int(payload["answers"][0]["answer_id"]))
            manifest_digests.add(
                str(payload["context_provenance"]["all_record_ids_sha256"])
            )
            seen_names.update(
                str(entity["canonical_name"]) for entity in payload["entity_catalog"]
            )
        self.assertEqual(len(request_ids), len(set(request_ids)))
        self.assertEqual(len(manifest_digests), 1)
        self.assertEqual(
            seen_names,
            {"Competitor 79"},
        )
        provenance = request_payloads[0][1]["context_provenance"]
        self.assertEqual(provenance["all_record_count"], 89)
        self.assertLess(provenance["selected_record_count"], 12)
        self.assertGreater(provenance["omitted_irrelevant_record_count"], 70)
        self.assertEqual(len(saved_annotations), 1)
        self.assertEqual(saved_annotations[0]["answer_id"], 1)

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
                "app.services.analyzer._prepare_analysis_foundation",
                new=AsyncMock(
                    return_value=(
                        {"score": 90},
                        {"findings": []},
                        {"brand_name": "Example"},
                        {"requested_site": {}},
                    )
                ),
            ),
            patch(
                "app.services.analyzer._save_answer_set_receipt",
                new=AsyncMock(),
            ),
            patch(
                "app.services.analyzer._seal_or_validate_panel_corpus_receipt",
                new=AsyncMock(),
            ),
            patch(
                "app.services.analyzer._validate_panel_foundation_resume",
                new_callable=AsyncMock,
            ) as validate_resume,
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
        validate_resume.assert_awaited_once()
        self.assertFalse(finish.await_args.kwargs["regenerate_illustrations"])
        generate_prompts.assert_not_awaited()
        run_panel.assert_not_awaited()

    async def test_saved_reprocess_seals_raw_corpus_before_any_analysis_write(
        self,
    ) -> None:
        events: list[str] = []

        async def seal(*_args: object, **_kwargs: object) -> dict[str, object]:
            events.append("seal")
            return {}

        async def progress(*_args: object, **_kwargs: object) -> None:
            events.append("progress")

        async def prepare(*_args: object, **_kwargs: object) -> tuple[object, ...]:
            events.append("prepare")
            return (
                {"score": 90},
                {"findings": []},
                {"brand_name": "Example"},
                {"requested_site": {}},
            )

        with (
            patch(
                "app.services.analyzer._seal_or_validate_panel_corpus_receipt",
                new=seal,
            ),
            patch("app.services.analyzer.update_progress", new=progress),
            patch(
                "app.services.analyzer._prepare_analysis_foundation",
                new=prepare,
            ),
            patch(
                "app.services.analyzer._validate_panel_foundation_resume",
                new=AsyncMock(),
            ),
            patch(
                "app.services.analyzer._save_answer_set_receipt",
                new=AsyncMock(),
            ),
            patch(
                "app.services.analyzer._finish_saved_answer_analysis",
                new=AsyncMock(),
            ),
        ):
            await reprocess_saved_answers("run-id")

        self.assertEqual(events[:3], ["seal", "progress", "prepare"])

    async def test_saved_reprocess_reports_grounding_block_without_panel_retry(
        self,
    ) -> None:
        failure = AsyncMock()
        panel = AsyncMock()
        refresh = AsyncMock(
            side_effect=OfferCatalogAdmissionError(
                "zero offer admission still not proven after refresh"
            )
        )
        with (
            patch(
                "app.services.analyzer.update_progress",
                new_callable=AsyncMock,
            ),
            patch(
                "app.services.analyzer._seal_or_validate_panel_corpus_receipt",
                new=AsyncMock(),
            ),
            patch(
                "app.services.analyzer._prepare_analysis_foundation",
                new=AsyncMock(
                    side_effect=OfferCatalogAdmissionError(
                        "zero offer admission not proven"
                    )
                ),
            ),
            patch("app.services.analyzer._run_panel", new=panel),
            patch(
                "app.services.analyzer._refresh_saved_reprocess_source_foundation",
                new=refresh,
            ),
            patch("app.services.analyzer.fail_run", new=failure),
        ):
            await reprocess_saved_answers("run-id")

        panel.assert_not_awaited()
        refresh.assert_awaited_once_with("run-id")
        failure.assert_awaited_once()
        message = failure.await_args.args[1]
        self.assertIn("проверка источников", message)
        self.assertIn("не повторный опрос моделей", message)
        self.assertEqual(
            failure.await_args.kwargs,
            {"failure_stage": "source_review_required"},
        )

    async def test_saved_reprocess_refreshes_sources_once_then_retries_without_panel(
        self,
    ) -> None:
        attempt = AsyncMock(
            side_effect=[
                crawler.CrawlAdmissionIncomplete("legacy crawl is incomplete"),
                None,
            ]
        )
        refresh = AsyncMock()
        panel = AsyncMock()
        failure = AsyncMock()
        with (
            patch(
                "app.services.analyzer._seal_or_validate_panel_corpus_receipt",
                new=AsyncMock(),
            ),
            patch("app.services.analyzer.update_progress", new=AsyncMock()),
            patch(
                "app.services.analyzer._run_saved_answer_analysis_attempt",
                new=attempt,
            ),
            patch(
                "app.services.analyzer._refresh_saved_reprocess_source_foundation",
                new=refresh,
            ),
            patch("app.services.analyzer._run_panel", new=panel),
            patch("app.services.analyzer.fail_run", new=failure),
        ):
            await reprocess_saved_answers("run-id")

        self.assertEqual(attempt.await_count, 2)
        self.assertFalse(attempt.await_args_list[0].kwargs["source_refresh_rebind"])
        self.assertTrue(attempt.await_args_list[1].kwargs["source_refresh_rebind"])
        refresh.assert_awaited_once_with("run-id")
        panel.assert_not_awaited()
        failure.assert_not_awaited()

    async def test_saved_reprocess_source_refresh_is_bounded_after_retry_failure(
        self,
    ) -> None:
        attempt = AsyncMock(
            side_effect=[
                OfferCatalogAdmissionError("catalog needs fresh sources"),
                OfferCatalogAdmissionError("catalog remains ungrounded"),
            ]
        )
        refresh = AsyncMock()
        panel = AsyncMock()
        failure = AsyncMock()
        with (
            patch(
                "app.services.analyzer._seal_or_validate_panel_corpus_receipt",
                new=AsyncMock(),
            ),
            patch("app.services.analyzer.update_progress", new=AsyncMock()),
            patch(
                "app.services.analyzer._run_saved_answer_analysis_attempt",
                new=attempt,
            ),
            patch(
                "app.services.analyzer._refresh_saved_reprocess_source_foundation",
                new=refresh,
            ),
            patch("app.services.analyzer._run_panel", new=panel),
            patch("app.services.analyzer.fail_run", new=failure),
        ):
            await reprocess_saved_answers("run-id")

        self.assertEqual(attempt.await_count, 2)
        refresh.assert_awaited_once_with("run-id")
        panel.assert_not_awaited()
        failure.assert_awaited_once()
        self.assertEqual(
            failure.await_args.kwargs,
            {"failure_stage": "source_review_required"},
        )

    async def test_saved_source_refresh_runs_only_the_shared_technical_executor(
        self,
    ) -> None:
        scope = {
            "version": "test",
            "run_id": "run-id",
            "owner": "operator-reprocess:test",
        }
        runtime = {
            "domain": "example.com",
            "page_limit": 8,
            "timeout_seconds": 20,
            "concurrency": 6,
            "user_agents": ["GPTBot", "Chrome-control"],
        }
        technical = AsyncMock(
            return_value={
                "pages": [("https://example.com/", "home")],
                "crawl_admission": {"admission_sha256": "a" * 64},
                "technical_matrix": {"receipt_sha256": "b" * 64},
                "site_preview": {"state": "captured"},
            }
        )
        panel = AsyncMock()
        save = AsyncMock()
        with (
            patch(
                "app.services.analyzer._claim_saved_source_refresh",
                new=AsyncMock(return_value=(scope, runtime, True)),
            ),
            patch(
                "app.services.analyzer.refresh_technical_foundation",
                new=technical,
            ),
            patch("app.services.analyzer._save_artifact", new=save),
            patch("app.services.analyzer._run_panel", new=panel),
        ):
            await _refresh_saved_reprocess_source_foundation("run-id")

        technical.assert_awaited_once()
        self.assertTrue(technical.await_args.kwargs["force_refresh"])
        self.assertEqual(
            technical.await_args.kwargs["progress_status"],
            RunStatus.analyzing,
        )
        panel.assert_not_awaited()
        save.assert_awaited_once()
        self.assertEqual(save.await_args.kwargs["status"], "completed")

    async def test_new_analysis_does_not_offer_retry_for_grounding_block(
        self,
    ) -> None:
        failure = AsyncMock()
        market = AsyncMock()
        panel = AsyncMock()
        with (
            patch(
                "app.services.analyzer.update_progress",
                new_callable=AsyncMock,
            ),
            patch(
                "app.services.analyzer.apply_ua_conditional_block",
                new_callable=AsyncMock,
            ),
            patch(
                "app.services.analyzer._load_panel_resume_checkpoint",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "app.services.analyzer._prepare_analysis_foundation",
                new=AsyncMock(
                    side_effect=OfferCatalogAdmissionError(
                        "zero offer admission not proven"
                    )
                ),
            ),
            patch("app.services.analyzer._market_research", new=market),
            patch("app.services.analyzer._run_panel", new=panel),
            patch("app.services.analyzer.fail_run", new=failure),
        ):
            await analyze_run("run-id")

        market.assert_not_awaited()
        panel.assert_not_awaited()
        failure.assert_awaited_once()
        message = failure.await_args.args[1]
        self.assertIn("проверка источников", message)
        self.assertNotIn("Продолжить", message)
        self.assertEqual(
            failure.await_args.kwargs,
            {"failure_stage": "source_review_required"},
        )

    async def test_visual_concepts_use_strong_model_with_independent_artifact(
        self,
    ) -> None:
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
                "app.services.analyzer._prepare_final_model_payload",
                new_callable=AsyncMock,
                return_value=(
                    {"prepared_complete_context": True},
                    {"mode": "direct", "coverage_complete": True},
                ),
            ),
            patch(
                "app.services.analyzer.chat_continuable_structured",
                new_callable=AsyncMock,
                return_value=response,
            ) as chat_mock,
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ) as save_artifact,
            patch(
                "app.services.analyzer._edit_illustration_copy_language",
                new=AsyncMock(return_value=concepts["illustrations"]),
            ),
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
        self.assertNotIn("max_completion_tokens", request)
        self.assertNotIn("max_tokens", request)
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
        self.assertTrue(payload["quality_requirements"]["no_unsupported_assertions"])
        self.assertTrue(
            content[1]["image_url"]["url"].startswith("data:image/png;base64,")
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
            call.kwargs["input_json"] for call in artifact_output.await_args_list
        ]
        request_inputs = [
            json.loads(call.kwargs["messages"][1]["content"][0]["text"])
            for call in chat_mock.await_args_list
        ]
        expected_hashes = [
            hashlib.sha256(prompt.encode("utf-8")).hexdigest() for prompt in prompts
        ]
        self.assertEqual(
            [payload["generation_prompt_sha256"] for payload in cache_inputs],
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
            image, review, accepted_prompt, attempts = await _generate_reviewed_image(
                "run-id",
                sequence=3,
                concept={"role": "web_memory_gap"},
                fact_context={"knowledge_gap": 0},
                base_prompt="Build a literal comparison.",
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
            image, review, _accepted_prompt, attempts = await _generate_reviewed_image(
                "run-id",
                sequence=1,
                concept={"role": "technical_access"},
                fact_context={"technical": {"score": 95}},
                base_prompt="Show server-readable content.",
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
            generated_names = sorted(path.name for path in generated.iterdir())
            self.assertEqual(len(generated_names), 3)
            self.assertTrue(generated_names[0].startswith("01-"))
            self.assertTrue(generated_names[1].startswith("02-"))
            self.assertTrue(generated_names[2].startswith("03-"))
            self.assertTrue(all(name.endswith(".png") for name in generated_names))

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
    def test_saved_bitmap_must_belong_to_the_same_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            static_dir = Path(tmp) / "static"
            generated_dir = static_dir / "generated"
            run_dir = generated_dir / "run-id"
            other_dir = generated_dir / "other-run"
            run_dir.mkdir(parents=True)
            other_dir.mkdir(parents=True)
            owned = run_dir / "01.png"
            foreign = other_dir / "01.png"
            owned.write_bytes(b"owned")
            foreign.write_bytes(b"foreign")

            with (
                patch("app.services.analyzer.STATIC_DIR", static_dir),
                patch("app.services.analyzer.GENERATED_DIR", generated_dir),
            ):
                self.assertEqual(
                    _saved_illustration_file_path(
                        "run-id",
                        "/static/generated/run-id/01.png",
                    ),
                    owned.resolve(),
                )
                self.assertIsNone(
                    _saved_illustration_file_path(
                        "run-id",
                        "/static/generated/other-run/01.png",
                    )
                )
                self.assertIsNone(
                    _saved_illustration_file_path(
                        "run-id",
                        "/static/generated/run-id/missing.png",
                    )
                )

    async def test_incomplete_illustration_editorial_receipt_blocks_publication(
        self,
    ) -> None:
        concepts = [
            {
                "role": "technical_access",
                "core_claim": "Сайт отдаёт основной текст.",
                "title": "Как сайт читают ИИ-системы",
                "caption": "Основной текст доступен в HTML.",
                "alt_text": "Схема доступа к тексту сайта.",
                "evidence_paths": ["/technical/score"],
            }
        ]
        edited_document = {
            "illustrations": [
                {
                    "role": "technical_access",
                    "core_claim": "Сайт отдаёт основной текст.",
                    "title": "Как сайт читают ИИ-системы",
                    "caption": "Основной текст доступен в HTML.",
                    "alt_text": "Схема доступа к тексту сайта.",
                    "evidence_paths": ["/technical/score"],
                }
            ]
        }
        with (
            patch(
                "app.services.analyzer._artifact_output",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "app.services.analyzer.edit_report",
                new=AsyncMock(
                    return_value=(
                        edited_document,
                        {
                            "coverage_complete": False,
                            "fallback_units": [{"unit_id": "x"}],
                        },
                    )
                ),
            ),
            patch(
                "app.services.analyzer._validate_illustration_concepts",
                return_value=[],
            ),
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ) as save_artifact,
        ):
            with self.assertRaisesRegex(
                OpenRouterError,
                "complete editorial contract",
            ):
                await _edit_illustration_copy_language(
                    "run-id",
                    concepts=concepts,
                    public_report={"technical": {"score": 100}},
                )

        terminal_writes = [
            call.kwargs
            for call in save_artifact.await_args_list
            if call.kwargs.get("artifact_key") == "illustration_copy_editorial"
            and call.kwargs.get("status") in {"completed", "failed"}
        ]
        self.assertEqual(len(terminal_writes), 1)
        self.assertEqual(terminal_writes[0]["status"], "failed")

    def test_reuse_refreshes_all_reader_copy_and_keeps_only_bitmap(self) -> None:
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
                    "alt_text": "Иллюстрация с пометкой о недостатке данных.",
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

    def test_reused_bitmap_gap_keeps_the_matching_concept_sequence(self) -> None:
        refreshed = [
            {"title": "Концепция 1"},
            {"title": "Концепция 2"},
            {"title": "Концепция 3"},
        ]
        public_rows = _reuse_saved_illustration_assets(
            [
                {
                    "sequence": 2,
                    "file_url": "/static/generated/run-id/02.png",
                }
            ],
            refreshed,
        )

        jobs = _reused_illustration_validation_jobs(public_rows, refreshed)

        self.assertEqual(len(jobs), 1)
        position, public_item, concept = jobs[0]
        self.assertEqual(position, 1)
        self.assertEqual(public_item["sequence"], 2)
        self.assertEqual(concept["title"], "Концепция 2")

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
            patch(
                "app.services.analyzer._revalidate_reused_illustration_assets",
                new=AsyncMock(
                    return_value=[
                        {
                            "sequence": 1,
                            "title": "Актуальный вывод",
                            "caption": "Подпись из пересчитанных метрик.",
                            "alt_text": "Актуальное описание.",
                            "file_url": "/static/generated/run-id/01.png",
                        }
                    ]
                ),
            ) as revalidate,
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
        revalidate.assert_awaited_once()
        generate.assert_not_awaited()

    async def test_reanalysis_omits_saved_images_when_copy_is_not_publishable(
        self,
    ) -> None:
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
                new=AsyncMock(side_effect=OpenRouterError("unsupported number 100")),
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
        self.assertEqual(reused, [])
        degraded_writes = [
            call.kwargs
            for call in save_artifact.await_args_list
            if call.kwargs.get("artifact_key") == "illustration_layer_degraded"
        ]
        self.assertEqual(len(degraded_writes), 1)
        self.assertEqual(degraded_writes[0]["status"], "completed")
        self.assertFalse(degraded_writes[0]["output_json"]["illustrations_published"])
        self.assertEqual(degraded_writes[0]["output_json"]["saved_asset_count"], 3)


class IllustrationPublicationSubsetTests(unittest.IsolatedAsyncioTestCase):
    async def test_file_and_qa_receipts_accept_one_or_two_of_three_roles(
        self,
    ) -> None:
        await init_db()
        review = {
            "usable": True,
            "facts_grounded": True,
            "claim_readable": True,
            "scores": {"context_specificity": 5},
            "unsupported_assertions": [],
            "hard_blockers": [],
            "visible_text_problems": [],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            static_dir = Path(temporary_directory) / "static"
            generated_dir = static_dir / "generated"
            with (
                patch("app.services.analyzer.STATIC_DIR", static_dir),
                patch("app.services.analyzer.GENERATED_DIR", generated_dir),
            ):
                for selected_sequences in ((2,), (1, 3)):
                    run_id = f"subset-{uuid.uuid4()}"
                    run_dir = generated_dir / run_id
                    run_dir.mkdir(parents=True)
                    public_rows: list[dict[str, Any]] = []
                    async with SessionLocal() as session:
                        session.add(
                            Run(
                                id=run_id,
                                domain="subset.example",
                                status=RunStatus.analyzing,
                                config_json={},
                            )
                        )
                        await session.flush()
                        for sequence in selected_sequences:
                            content = f"image-{sequence}".encode()
                            image_sha256 = hashlib.sha256(content).hexdigest()
                            filename = f"{sequence:02d}-{image_sha256}.png"
                            (run_dir / filename).write_bytes(content)
                            file_url = f"/static/generated/{run_id}/{filename}"
                            public_rows.append(
                                {
                                    "sequence": sequence,
                                    "title": f"Схема {sequence}",
                                    "caption": "Подпись",
                                    "alt_text": "Описание",
                                    "file_url": file_url,
                                }
                            )
                            session.add(
                                ReportIllustration(
                                    run_id=run_id,
                                    sequence=sequence,
                                    title=f"Схема {sequence}",
                                    caption="Подпись",
                                    alt_text="Описание",
                                    file_url=file_url,
                                    generation_prompt="prompt",
                                    model="test/image",
                                    usage_json={
                                        "image_sha256": image_sha256,
                                        "quality_version": ILLUSTRATION_QA_VERSION,
                                        "quality_model": PROCESSING_MODEL,
                                        "quality_review": review,
                                    },
                                )
                            )
                            session.add(
                                RunArtifact(
                                    run_id=run_id,
                                    stage_key="report",
                                    artifact_key=(
                                        f"illustration_qa_{sequence}_"
                                        f"{image_sha256[:16]}"
                                    ),
                                    status="completed",
                                    model=PROCESSING_MODEL,
                                    prompt_version=ILLUSTRATION_QA_VERSION,
                                    input_json={"image_sha256": image_sha256},
                                    output_json=review,
                                )
                            )
                        await session.commit()
                    try:
                        receipts = await _verified_illustration_asset_receipts(
                            run_id,
                            public_rows,
                        )
                        self.assertEqual(
                            [row["sequence"] for row in receipts],
                            list(selected_sequences),
                        )
                        self.assertTrue(all(row["qa_verified"] for row in receipts))
                    finally:
                        async with SessionLocal() as session:
                            await session.execute(delete(Run).where(Run.id == run_id))
                            await session.commit()


class OptionalReportAssetAdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def test_quarantines_only_broken_optional_assets_and_seals_receipt(
        self,
    ) -> None:
        valid = {
            "sequence": 1,
            "title": "Рабочая схема",
            "caption": "Подпись",
            "alt_text": "Описание",
            "file_url": "/static/generated/run-id/01.png",
        }
        broken = {
            "sequence": 2,
            "title": "Повреждённая схема",
            "caption": "Подпись",
            "alt_text": "Описание",
            "file_url": "/static/generated/run-id/02.png",
        }
        preview = {"file_url": "/static/generated/run-id/preview.jpg"}
        report_json = {
            "narrative": {"headline": "Готовый аналитический отчёт"},
            "illustrations": [valid, broken],
            "site_preview": preview,
        }

        async def verify(_run_id: str, rows: list[dict[str, Any]]) -> list[dict]:
            if rows[0]["sequence"] == 2:
                raise PublicationContractError("QA receipt is missing")
            return [{"sequence": 1, "qa_verified": True}]

        with (
            patch(
                "app.services.analyzer._verified_illustration_asset_receipts",
                new=AsyncMock(side_effect=verify),
            ),
            patch(
                "app.services.analyzer.site_preview_asset_receipt",
                side_effect=ValueError("site_preview_asset_invalid"),
            ),
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ) as save_artifact,
        ):
            sanitized, illustrations, receipt = (
                await _sanitize_optional_report_assets(
                    "run-id",
                    report_json=report_json,
                    illustrations=[valid, broken],
                )
            )

        self.assertEqual(illustrations, [valid])
        self.assertEqual(sanitized["illustrations"], [valid])
        self.assertNotIn("site_preview", sanitized)
        self.assertEqual(report_json["illustrations"], [valid, broken])
        self.assertIn("site_preview", report_json)
        self.assertEqual(receipt["state"], "degraded")
        self.assertTrue(receipt["degraded"])
        self.assertEqual(
            receipt["reason_codes"],
            [
                "illustration_asset_or_qa_receipt_invalid",
                "site_preview_asset_invalid_or_missing",
            ],
        )
        self.assertEqual(receipt["illustrations"]["published_sequences"], [1])
        self.assertEqual(receipt["illustrations"]["rejected_count"], 1)
        self.assertFalse(receipt["site_preview"]["published"])
        receipt_core = {
            key: value
            for key, value in receipt.items()
            if key not in {"artifact_key", "receipt_sha256"}
        }
        self.assertEqual(receipt["receipt_sha256"], _stable_json_sha256(receipt_core))
        self.assertEqual(
            receipt["artifact_key"],
            OPTIONAL_ASSET_ADMISSION_PREFIX + receipt["receipt_sha256"],
        )
        save_artifact.assert_awaited_once()
        saved = save_artifact.await_args.kwargs
        self.assertEqual(saved["status"], "completed")
        self.assertEqual(saved["output_json"], receipt)

    async def test_report_illustration_snapshot_mismatch_remains_hard_failure(
        self,
    ) -> None:
        public_row = {
            "sequence": 1,
            "file_url": "/static/generated/run-id/01.png",
        }
        with (
            patch(
                "app.services.analyzer._verified_illustration_asset_receipts",
                new_callable=AsyncMock,
            ) as verify,
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ) as save_artifact,
        ):
            with self.assertRaises(PublicationContractError):
                await _sanitize_optional_report_assets(
                    "run-id",
                    report_json={"illustrations": []},
                    illustrations=[public_row],
                )

        verify.assert_not_awaited()
        save_artifact.assert_not_awaited()

    async def test_missing_preview_degrades_without_blocking_report(self) -> None:
        report_json = {
            "narrative": {"headline": "Готовый аналитический отчёт"},
            "illustrations": [],
        }
        with (
            patch(
                "app.services.analyzer._verified_illustration_asset_receipts",
                new_callable=AsyncMock,
            ) as verify,
            patch(
                "app.services.analyzer.site_preview_asset_receipt",
            ) as preview_receipt,
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ) as save_artifact,
        ):
            sanitized, illustrations, receipt = (
                await _sanitize_optional_report_assets(
                    "run-id",
                    report_json=report_json,
                    illustrations=[],
                )
            )

        self.assertEqual(sanitized, report_json)
        self.assertEqual(illustrations, [])
        self.assertEqual(receipt["state"], "degraded")
        self.assertEqual(receipt["reason_codes"], ["site_preview_asset_missing"])
        self.assertEqual(
            receipt["site_preview"],
            {
                "requested": True,
                "published": False,
                "reason_codes": ["site_preview_asset_missing"],
            },
        )
        verify.assert_not_awaited()
        preview_receipt.assert_not_called()
        save_artifact.assert_awaited_once()


class ReaderCopyManifestTests(unittest.TestCase):
    @staticmethod
    def _accepted_receipts() -> dict[str, dict[str, object]]:
        return {
            "final_report": {"accepted": True, "reasons": []},
            "technical_review": {"accepted": True, "reasons": []},
            "illustrations": {"accepted": True, "reasons": []},
        }

    def test_gate_degrades_copy_quality_but_blocks_snapshot_or_receipt_tamper(
        self,
    ) -> None:
        publication = {"blocking_reasons": []}
        clean_lint = {
            "blocking": False,
            "omitted_issue_count": 0,
            "issues": [],
        }
        accepted = _reader_copy_gate_decision(
            publication=publication,
            lint=clean_lint,
            receipts=self._accepted_receipts(),
        )
        self.assertEqual(accepted["decision"], "pass")
        self.assertTrue(accepted["quality_complete"])

        lint_degraded = _reader_copy_gate_decision(
            publication=publication,
            lint={
                **clean_lint,
                "blocking": True,
                "issues": [{"code": "long_dash"}],
            },
            receipts=self._accepted_receipts(),
        )
        self.assertEqual(lint_degraded["decision"], "degraded_safe")
        self.assertEqual(lint_degraded["blocking_reasons"], [])
        self.assertIn("copy_lint_findings", lint_degraded["degraded_reasons"])

        truncated_lint = _reader_copy_gate_decision(
            publication=publication,
            lint={
                **clean_lint,
                "omitted_issue_count": 9,
            },
            receipts=self._accepted_receipts(),
        )
        self.assertEqual(truncated_lint["decision"], "degraded_safe")
        self.assertIn(
            "copy_lint_not_exhaustive",
            truncated_lint["degraded_reasons"],
        )

        receipts = self._accepted_receipts()
        receipts["technical_review"] = {
            "accepted": False,
            "reasons": ["quality_incomplete"],
        }
        receipt_degraded = _reader_copy_gate_decision(
            publication=publication,
            lint=clean_lint,
            receipts=receipts,
        )
        self.assertEqual(receipt_degraded["decision"], "degraded_safe")
        self.assertIn(
            "technical_review_receipt_incomplete",
            receipt_degraded["degraded_reasons"],
        )
        self.assertEqual(receipt_degraded["blocking_reasons"], [])

        receipts = self._accepted_receipts()
        receipts["optional_assets"] = {
            "accepted": False,
            "reasons": ["site_preview_asset_missing"],
        }
        optional_asset_degraded = _reader_copy_gate_decision(
            publication=publication,
            lint=clean_lint,
            receipts=receipts,
        )
        self.assertEqual(optional_asset_degraded["decision"], "degraded_safe")
        self.assertIn(
            "optional_assets_receipt_incomplete",
            optional_asset_degraded["degraded_reasons"],
        )
        self.assertEqual(optional_asset_degraded["blocking_reasons"], [])

        receipts["technical_review"] = {
            "accepted": False,
            "reasons": ["published_digest_mismatch"],
        }
        receipt_tamper = _reader_copy_gate_decision(
            publication=publication,
            lint=clean_lint,
            receipts=receipts,
        )
        self.assertEqual(receipt_tamper["decision"], "block")
        self.assertIn(
            "technical_review:published_digest_mismatch",
            receipt_tamper["blocking_reasons"],
        )

        snapshot_mismatch = _reader_copy_gate_decision(
            publication={"blocking_reasons": ["report_json_snapshot_mismatch"]},
            lint=clean_lint,
            receipts=self._accepted_receipts(),
        )
        self.assertEqual(snapshot_mismatch["decision"], "block")

    def test_illustration_copy_receipt_accepts_verified_one_or_two_row_subset(
        self,
    ) -> None:
        audited_rows = [
            {
                "role": f"role-{sequence}",
                "core_claim": f"Факт {sequence}",
                "title": f"Заголовок {sequence}",
                "caption": f"Подпись {sequence}",
                "alt_text": f"Описание {sequence}",
                "evidence_paths": [],
            }
            for sequence in range(1, 4)
        ]
        audited_document = {"illustrations": audited_rows}

        async def editor(payload: dict[str, Any]) -> dict[str, Any]:
            edited = str(payload["core_text"])
            return {
                "source_unit_id": payload["source_unit_id"],
                "source_sha256": payload["source_sha256"],
                "edited_text": edited,
                "claim_receipts": [
                    {
                        "claim_sha256": claim["claim_sha256"],
                        "preserved": True,
                        "target_excerpt": claim["source_excerpt"],
                        "note": "Смысл сохранён.",
                    }
                    for claim in payload["source_claims"]
                ],
                "new_claims": [],
            }

        async def critic(payload: dict[str, Any]) -> dict[str, Any]:
            return {
                "verdict": "pass",
                "issues": [],
                "claim_checks": [
                    {
                        "claim_sha256": claim["claim_sha256"],
                        "meaning_preserved": True,
                        "actor_preserved": True,
                        "scope_preserved": True,
                        "numbers_preserved": True,
                        "actor_or_mechanism_explicit": True,
                        "number_carrier_explicit": True,
                        "active_voice": True,
                        "no_slogan_or_meta": True,
                        "no_mechanical_triad": True,
                        "reason": "Смысл совпадает.",
                    }
                    for claim in payload["source_claims"]
                ],
                "new_claims": [],
            }

        edited_document, audit = asyncio.run(
            edit_report(
                audited_document,
                editor_call=editor,
                critic_call=critic,
                prose_paths=illustration_copy_narrative_paths(audited_document),
            )
        )
        self.assertEqual(edited_document, audited_document)
        self.assertTrue(audit["quality_complete"])
        artifact = RunArtifact(
            run_id="run-id",
            stage_key="report",
            artifact_key="illustration_copy_editorial",
            status="completed",
            prompt_version="test",
            output_json={"copy": audited_document, "audit": audit},
        )

        for sequences in ((2,), (1, 3)):
            with self.subTest(sequences=sequences):
                published = [
                    {
                        "sequence": sequence,
                        "title": audited_rows[sequence - 1]["title"],
                        "caption": audited_rows[sequence - 1]["caption"],
                        "alt_text": audited_rows[sequence - 1]["alt_text"],
                        "file_url": f"/static/generated/run-id/{sequence}.png",
                    }
                    for sequence in sequences
                ]
                receipt = _illustration_receipt_state(
                    artifact,
                    published,
                    source_document=audited_document,
                    prose_paths=illustration_copy_narrative_paths(audited_document),
                    source_artifact_key="illustration_concepts",
                )

                self.assertTrue(receipt["accepted"])
                self.assertEqual(receipt["published_count"], len(sequences))
                self.assertEqual(receipt["audited_count"], 3)
                self.assertEqual(receipt["published_sequences"], list(sequences))

        zero = _illustration_receipt_state(None, [])
        self.assertTrue(zero["accepted"])
        self.assertTrue(zero["publication_policy"]["zero_assets_allowed"])

    def test_registry_covers_dynamic_report_copy_beyond_the_final_narrative(
        self,
    ) -> None:
        final = {
            "headline": "Сайт доступен моделям",
            "headline_emphasis": [],
            "verdict": "Сервер отдаёт основной текст.",
            "executive_summary": "Бренд встречается в части ответов.",
            "sections": [{"heading": "Вывод", "body": "Нужна разметка."}],
            "actions": [
                {
                    "priority": "now",
                    "title": "Добавить сущности",
                    "why": "Моделям не хватает связи.",
                    "step": "Опубликовать Schema.org.",
                    "evidence": "Дословное доказательство.",
                }
            ],
            "limitations": ["Оценка относится к проверенным страницам."],
        }
        document = _reader_copy_document(
            final_report=final,
            public_report={
                "brand": {
                    "site_type": "service",
                    "category": "Маркетинг",
                    "positioning": "Агентство performance-маркетинга.",
                },
                "technical": {
                    "summary": {
                        "facts": [
                            {
                                "label": "Содержательные страницы",
                                "detail": "Проверены восемь страниц сайта.",
                            }
                        ]
                    },
                    "barriers": [
                        {
                            "title": "Не хватает описания сущностей",
                            "detail": "Schema.org не объясняет продукты сайта.",
                        }
                    ],
                    "review": {
                        "overall_conclusion": "Основной текст доступен.",
                        "render_conclusion": "JavaScript не обязателен.",
                        "findings": [
                            {
                                "title": "Не хватает разметки",
                                "severity": "medium",
                                "evidence": "Сущности — не связаны разметкой.",
                                "business_effect": "Модели теряют контекст.",
                                "action": "Добавить Schema.org.",
                            }
                        ],
                        "limitations": [],
                    },
                },
                "methodology": {
                    "summary": "Метрики рассчитаны из доказательной разметки.",
                    "modes": [
                        {
                            "name": "С веб-поиском",
                            "description": "Модели используют актуальные источники.",
                        }
                    ],
                    "offline_limit": "Perplexity не входит в срез памяти.",
                },
                "key_metrics": {
                    "technical_access": {
                        "label": "Техническая готовность проверенного среза",
                        "unit": "/ 100",
                        "coverage_label": "Ограниченный срез",
                    }
                },
            },
            illustrations=[
                {
                    "sequence": 1,
                    "title": "Как сайт читают модели",
                    "caption": "Основной текст доступен в HTML.",
                    "alt_text": "Схема доступа к тексту сайта.",
                    "file_url": "/static/generated/run-id/01.png",
                }
            ],
        )

        self.assertEqual(document["final_report"], final)
        self.assertEqual(
            document["action_basis_copy"],
            ["Дословное доказательство."],
        )
        self.assertEqual(
            document["technical_finding_basis_copy"],
            ["Сущности — не связаны разметкой."],
        )
        lint = lint_reader_copy_tree(document)
        self.assertTrue(lint.blocking)
        self.assertIn(
            "$.technical_finding_basis_copy[0]",
            {issue.path for issue in lint.issues},
        )
        self.assertEqual(
            document["published_report"]["technical"]["summary"]["facts"][0]["detail"],
            "Проверены восемь страниц сайта.",
        )
        self.assertEqual(
            document["published_report"]["technical"]["barriers"][0]["title"],
            "Не хватает описания сущностей",
        )
        self.assertEqual(
            document["technical_review"]["overall_conclusion"],
            "Основной текст доступен.",
        )
        self.assertEqual(
            document["methodology"]["modes"][0]["description"],
            "Модели используют актуальные источники.",
        )
        self.assertEqual(
            document["metric_labels"]["technical_access"]["coverage_label"],
            "Ограниченный срез",
        )
        self.assertEqual(
            document["illustrations"][0]["alt_text"],
            "Схема доступа к тексту сайта.",
        )
        self.assertNotIn("file_url", document["illustrations"][0])

    def test_publication_contract_binds_markdown_and_json_to_edited_copy(
        self,
    ) -> None:
        final = {
            "headline": "Сайт доступен моделям",
            "headline_emphasis": [],
            "verdict": "Сервер отдаёт основной текст.",
            "executive_summary": "Бренд встречается в части ответов.",
            "sections": [{"heading": "Вывод", "body": "Нужна разметка."}],
            "actions": [],
            "limitations": [],
        }
        public = {"brand": {"name": "Example"}, "technical": {"score": 90}}
        illustrations: list[dict[str, Any]] = []
        narrative = {
            "headline": final["headline"],
            "headline_emphasis": [],
            "verdict": final["verdict"],
            "executive_summary": final["executive_summary"],
            "actions": [],
        }
        report_json = {
            **public,
            "narrative": narrative,
            "illustrations": illustrations,
        }

        accepted = _reader_copy_publication_contract(
            final_report=final,
            public_report=public,
            illustrations=illustrations,
            analysis_markdown=_render_markdown(final),
            report_json=report_json,
        )
        self.assertEqual(accepted["blocking_reasons"], [])
        self.assertTrue(all(accepted["checks"].values()))

        rejected = _reader_copy_publication_contract(
            final_report=final,
            public_report=public,
            illustrations=illustrations,
            analysis_markdown="# Подменённый текст",
            report_json={**report_json, "narrative": {**narrative, "verdict": "0"}},
        )
        self.assertIn(
            "analysis_markdown_matches_final_report",
            rejected["blocking_reasons"],
        )
        self.assertIn(
            "report_json_narrative_matches_final_report",
            rejected["blocking_reasons"],
        )


class FinalAnswerCorpusTests(unittest.TestCase):
    def _rows(
        self,
        count: int = 13,
        *,
        last_suffix: str = "LAST-SENTINEL",
    ) -> list[tuple[ModelAnswer, VisibilityPrompt, AnswerAnnotation]]:
        rows: list[tuple[ModelAnswer, VisibilityPrompt, AnswerAnnotation]] = []
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
                role=("brand_diagnostic" if index >= 15 else "unbranded_discovery"),
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
                    "target_role": ("recommended" if index % 3 == 0 else "absent"),
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

        self.assertIsNone(FINAL_CONTEXT_MAX_ANSWERS)
        self.assertEqual(len(selected), len(items))
        self.assertEqual(selection_manifest["omitted_count"], 0)
        self.assertEqual(
            selection_manifest["policy"],
            "complete_attested_corpus_no_local_cap_v1",
        )
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
                self.assertTrue(item["answer_text"].endswith(("END", "LAST-SENTINEL")))
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
            (
                3,
                "p1",
                1,
                "web",
                "T",
                "b",
                "legacy",
                True,
                False,
                "recommended",
                "positive",
            ),
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
                "prompt_key": "u-1",
                "intent_class": "I",
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
            any("без веб-поиска" in item["name"] for item in methodology["modes"])
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

        public_knowledge_provider_memory = report["brand_knowledge"]["providers"][0][
            "memory"
        ]
        self.assertNotIn("facets", public_knowledge_provider_memory)
        self.assertTrue(
            public_knowledge_provider_memory["qualitative_context_withheld"]
        )
        public_discovery_provider_memory = report["discovery"]["providers"][0][
            "brand_knowledge"
        ]["memory"]
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
            [item["raw_answer_sha256"] for item in final_manifest["observed_cells"]],
            [item["provenance"]["raw_answer_sha256"] for item in corpus],
        )

    def test_final_context_preflight_is_explicit_and_never_truncates(self) -> None:
        payload = {"answer_corpus": [{"answer_text": "я" * 100}]}
        serialized_bytes = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))

        preflight = _final_input_preflight(payload, token_budget=1)

        self.assertEqual(preflight["state"], "direct")
        self.assertTrue(preflight["accepted"])
        self.assertFalse(preflight["legacy_budget_would_accept"])
        self.assertEqual(
            preflight["budget_enforcement"],
            "disabled_diagnostics_only",
        )
        self.assertEqual(
            preflight["estimated_input_tokens"],
            serialized_bytes,
        )

        estimated = preflight["estimated_input_tokens"]
        reserved = _final_input_preflight(
            payload,
            token_budget=estimated + 9,
            reserve_tokens=10,
        )
        self.assertTrue(reserved["accepted"])
        self.assertFalse(reserved["legacy_budget_would_accept"])
        self.assertEqual(reserved["repair_reserve_tokens"], 10)

    def test_unicode_payload_cannot_bypass_physical_token_window(self) -> None:
        payload = {"answer": "видимость" * 200}
        serialized_bytes = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))

        preflight = _final_input_preflight(
            payload,
            physical_input_token_window=serialized_bytes - 1,
        )

        self.assertEqual(preflight["estimated_input_tokens"], serialized_bytes)
        self.assertFalse(preflight["physical_direct_fit"])
        self.assertEqual(preflight["state"], "partition_required")
        self.assertEqual(
            preflight["estimation_contract"],
            "conservative_upper_bound:1_serialized_utf8_byte_per_token",
        )


class FinalInputHarnessTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _oversized_payload() -> dict[str, Any]:
        return {
            "report_data": {
                "critical_metric": 73,
                "data_state": "unknown",
                "eligible": True,
                "optional_value": None,
                "long_narrative": "x" * 260_000,
            }
        }

    @staticmethod
    def _window() -> dict[str, Any]:
        return {
            "version": "test",
            "resolution": "test",
            "model_envelope": {},
            "input_token_window": 192_000,
            "input_utf8_window": 192_000,
        }

    def test_final_input_mapper_schema_uses_provider_supported_keywords(
        self,
    ) -> None:
        unique_item_paths: list[str] = []
        incomplete_required_paths: list[str] = []
        duplicate_required_paths: list[str] = []

        def visit(value: Any, path: str = "") -> None:
            if isinstance(value, dict):
                properties = value.get("properties")
                if value.get("type") == "object" and isinstance(properties, dict):
                    required = value.get("required")
                    if not isinstance(required, list) or set(required) != set(
                        properties
                    ):
                        incomplete_required_paths.append(path or "/")
                    elif len(required) != len(set(required)):
                        duplicate_required_paths.append(path or "/")
                for key, child in value.items():
                    child_path = f"{path}/{key}"
                    if key == "uniqueItems":
                        unique_item_paths.append(child_path)
                    visit(child, child_path)
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    visit(child, f"{path}/{index}")

        visit(FINAL_INPUT_EVIDENCE_SCHEMA, "/mapper")
        visit(FINAL_INPUT_ROOT_SUMMARY_SCHEMA, "/root_summary")

        self.assertEqual(unique_item_paths, [])
        self.assertEqual(incomplete_required_paths, [])
        self.assertEqual(duplicate_required_paths, [])
        observation_properties = FINAL_INPUT_EVIDENCE_SCHEMA["properties"][
            "observations"
        ]["items"]["properties"]
        for field in ("source_paths", "source_unit_ids", "source_claim_ids"):
            self.assertEqual(observation_properties[field]["minItems"], 1)
            self.assertEqual(observation_properties[field]["maxItems"], 1)

    @staticmethod
    def _stable_sha(value: Any) -> str:
        return hashlib.sha256(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    @classmethod
    def _scalar_fact_binding(
        cls,
        *,
        source_path: str,
        json_literal: str,
        value_type: str = "string",
    ) -> dict[str, Any]:
        contract = {
            "contract": "code_owned_exact_scalar_binding",
            "source_path": source_path,
            "value_type": value_type,
            "json_literal": json_literal,
            "source_value_sha256": hashlib.sha256(
                json_literal.encode("utf-8")
            ).hexdigest(),
        }
        return {
            "binding_id": "afb_scalar_" + cls._stable_sha(contract),
            **contract,
        }

    @classmethod
    def _bounded_leaf(
        cls,
        *,
        node_id: str,
        index: int,
        semantic_text: str,
        claim_count: int,
        bindings: list[dict[str, Any]],
    ) -> dict[str, Any]:
        table = _final_root_fact_table(bindings)
        binding_ids = [str(binding["binding_id"]) for binding in bindings]
        fact_provenance_sha256 = cls._stable_sha(
            {
                "domain": "aiv-final-root-leaf-facts-v1",
                "mandatory_fact_bindings_sha256": cls._stable_sha(bindings),
                "mandatory_fact_ids_sha256": cls._stable_sha(binding_ids),
                "mandatory_fact_table_sha256": cls._stable_sha(table),
            }
        )
        model_view = {
            "source_node_id": node_id,
            "context_value": semantic_text,
            "mandatory_fact_count": len(bindings),
            "mandatory_fact_provenance_sha256": fact_provenance_sha256,
        }
        return {
            "node_id": node_id,
            "level": 0,
            "leaf_start": index,
            "leaf_end": index + 1,
            "descendant_source_node_count": 1,
            "descendant_claim_count": claim_count,
            "tree_commitment": f"leaf-commitment-{index}",
            "model_view": model_view,
            "source_text": json.dumps(model_view),
            "semantic_text": semantic_text,
            "mandatory_fact_bindings": copy.deepcopy(bindings),
        }

    @classmethod
    def _bounded_proof(cls, parent: dict[str, Any]) -> dict[str, Any]:
        return {
            "receipt": _final_root_node_receipt(parent),
            "ordered_child_receipts": parent["child_receipts"],
            "ordered_child_receipts_sha256": cls._stable_sha(parent["child_receipts"]),
            "packet_sha256": parent["packet_sha256"],
            "model_view": copy.deepcopy(parent["model_view"]),
            "mandatory_fact_count": parent["mandatory_fact_count"],
            "mandatory_fact_provenance_sha256": parent[
                "mandatory_fact_provenance_sha256"
            ],
            "fact_provenance": copy.deepcopy(parent["fact_provenance"]),
        }

    def test_structural_string_passthrough_is_not_length_gated(self) -> None:
        metric_code = "metric-code-" + ("x" * 2_000)
        narrative = "narrative-" + ("y" * 2_000)
        units, _manifest = _flatten_final_input_payload(
            {
                "report_data": {
                    "metric_code": metric_code,
                    "long_narrative": narrative,
                }
            },
            target_chars=4_096,
        )

        passthrough = {
            item["source_path"]: item
            for item in _final_input_deterministic_passthrough(units)
        }

        self.assertEqual(
            passthrough["/report_data/metric_code"]["value"],
            metric_code,
        )
        self.assertNotIn("/report_data/long_narrative", passthrough)

    async def test_structural_units_bypass_mapper_without_losing_coverage(
        self,
    ) -> None:
        payload = {
            "report_data": {
                "score": 73,
                "status": "completed",
                "eligible": True,
                "optional_value": None,
                "note": "Содержательный вывод клиента",
            }
        }
        mapper_payloads: list[dict[str, Any]] = []

        async def capture_mapper(*_args: Any, **kwargs: Any) -> dict[str, Any]:
            user_payload = kwargs["user_payload"]
            mapper_payloads.append(copy.deepcopy(user_payload))
            return self._packet_for_units(
                user_payload["source_units"],
                source_claims=user_payload.get("source_claims"),
            )

        with (
            patch(
                "app.services.analyzer._final_model_input_window",
                new_callable=AsyncMock,
                return_value=self._window(),
            ),
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ),
            patch(
                "app.services.analyzer._structured_artifact",
                new_callable=AsyncMock,
                side_effect=capture_mapper,
            ) as structured_artifact,
        ):
            model_payload, plan = await _prepare_final_model_payload(
                "run-id",
                payload=payload,
                system="author",
                force_hierarchical=True,
            )

        self.assertEqual(structured_artifact.await_count, 1)
        mapped_paths = {
            str(unit["source_path"])
            for mapper_payload in mapper_payloads
            for unit in mapper_payload["source_units"]
        }
        self.assertEqual(mapped_paths, {"/report_data/note"})
        passthrough = {
            item["source_path"]: item
            for item in model_payload["deterministic_passthrough"]["values"]
        }
        self.assertEqual(passthrough["/report_data/score"]["value"], 73)
        self.assertEqual(passthrough["/report_data/status"]["value"], "completed")
        self.assertIs(passthrough["/report_data/eligible"]["value"], True)
        self.assertIsNone(passthrough["/report_data/optional_value"]["value"])
        self.assertEqual(plan["mapped_source_unit_count"], 1)
        self.assertEqual(plan["code_owned_source_unit_count"], 4)
        self.assertEqual(plan["source_unit_count"], 5)
        self.assertEqual(plan["covered_claim_count"], plan["source_claim_count"])
        self.assertEqual(
            len(model_payload["evidence_digest"]["unit_coverage"]),
            plan["source_unit_count"],
        )

    async def test_all_structural_payload_needs_no_mapper_call(self) -> None:
        with (
            patch(
                "app.services.analyzer._final_model_input_window",
                new_callable=AsyncMock,
                return_value=self._window(),
            ),
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ),
            patch(
                "app.services.analyzer._structured_artifact",
                new_callable=AsyncMock,
            ) as structured_artifact,
        ):
            model_payload, plan = await _prepare_final_model_payload(
                "run-id",
                payload={
                    "report_data": {
                        "score": 73,
                        "eligible": True,
                        "optional_value": None,
                    }
                },
                system="author",
                force_hierarchical=True,
            )

        structured_artifact.assert_not_awaited()
        self.assertEqual(plan["map_leaf_count"], 0)
        self.assertEqual(plan["mapped_source_unit_count"], 0)
        self.assertEqual(plan["code_owned_source_unit_count"], 3)
        self.assertEqual(plan["terminal_reducer_mode"], "code_owned_scalar_only")
        self.assertEqual(
            len(model_payload["evidence_digest"]["unit_coverage"]),
            3,
        )

    async def test_systemic_map_failure_stops_at_canary(self) -> None:
        payload = {
            "report_data": {
                "notes": [f"Содержательный вывод {index}" for index in range(10)]
            }
        }

        def singleton_packs(
            units: list[dict[str, Any]],
            **_kwargs: Any,
        ) -> list[list[dict[str, Any]]]:
            return [[unit] for unit in units]

        with (
            patch(
                "app.services.analyzer._final_model_input_window",
                new_callable=AsyncMock,
                return_value=self._window(),
            ),
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ),
            patch(
                "app.services.analyzer._pack_final_input_units",
                side_effect=singleton_packs,
            ),
            patch(
                "app.services.analyzer._structured_artifact",
                new_callable=AsyncMock,
                side_effect=OpenRouterError("invalid_json_schema"),
            ) as structured_artifact,
        ):
            with self.assertRaisesRegex(OpenRouterError, "invalid_json_schema"):
                await _prepare_final_model_payload(
                    "run-id",
                    payload=payload,
                    system="author",
                    force_hierarchical=True,
                )

        self.assertEqual(structured_artifact.await_count, 1)

    async def test_failed_map_wave_never_starts_the_next_wave(self) -> None:
        payload = {
            "report_data": {
                "notes": [f"Содержательный вывод {index}" for index in range(10)]
            }
        }
        call_count = 0

        def singleton_packs(
            units: list[dict[str, Any]],
            **_kwargs: Any,
        ) -> list[list[dict[str, Any]]]:
            return [[unit] for unit in units]

        async def fail_first_wave(
            *_args: Any,
            **kwargs: Any,
        ) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise OpenRouterError("systemic-wave-error")
            user_payload = kwargs["user_payload"]
            return self._packet_for_units(
                user_payload["source_units"],
                source_claims=user_payload.get("source_claims"),
            )

        with (
            patch(
                "app.services.analyzer._final_model_input_window",
                new_callable=AsyncMock,
                return_value=self._window(),
            ),
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ),
            patch(
                "app.services.analyzer._pack_final_input_units",
                side_effect=singleton_packs,
            ),
            patch(
                "app.services.analyzer._structured_artifact",
                new_callable=AsyncMock,
                side_effect=fail_first_wave,
            ) as structured_artifact,
        ):
            with self.assertRaisesRegex(OpenRouterError, "systemic-wave-error"):
                await _prepare_final_model_payload(
                    "run-id",
                    payload=payload,
                    system="author",
                    force_hierarchical=True,
                )

        self.assertEqual(
            structured_artifact.await_count,
            1 + PROCESSING_BATCH_CONCURRENCY,
        )

    async def test_incomplete_mapper_pack_splits_without_losing_claims(
        self,
    ) -> None:
        payload = {
            "report_data": {
                "notes": [f"Содержательный вывод {index}" for index in range(4)]
            }
        }
        successful_mapper = self._successful_mapper()
        parent_sizes: list[int] = []

        def one_pack(
            units: list[dict[str, Any]],
            **_kwargs: Any,
        ) -> list[list[dict[str, Any]]]:
            return [units]

        async def partial_then_complete(
            *_args: Any,
            **kwargs: Any,
        ) -> dict[str, Any]:
            user_payload = kwargs["user_payload"]
            source_units = user_payload.get("source_units")
            if isinstance(source_units, list):
                parent_sizes.append(len(source_units))
                if len(source_units) > 2:
                    accepted_units = source_units[:2]
                    accepted_ids = {
                        str(unit["source_unit_id"]) for unit in accepted_units
                    }
                    accepted_claims = [
                        claim
                        for claim in user_payload.get("source_claims") or []
                        if str(claim["source_unit_id"]) in accepted_ids
                    ]
                    return self._packet_for_units(
                        accepted_units,
                        source_claims=accepted_claims,
                    )
            return await successful_mapper(*_args, **kwargs)

        with (
            patch(
                "app.services.analyzer._final_model_input_window",
                new_callable=AsyncMock,
                return_value=self._window(),
            ),
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ) as save_artifact,
            patch(
                "app.services.analyzer._artifact_output",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "app.services.analyzer._pack_final_input_units",
                side_effect=one_pack,
            ),
            patch(
                "app.services.analyzer._structured_artifact",
                new_callable=AsyncMock,
                side_effect=partial_then_complete,
            ),
        ):
            model_payload, plan = await _prepare_final_model_payload(
                "run-id",
                payload=payload,
                system="author",
                force_hierarchical=True,
            )

        self.assertEqual(parent_sizes[:3], [4, 2, 2])
        self.assertEqual(plan["covered_claim_count"], plan["source_claim_count"])
        self.assertEqual(
            model_payload["long_input_contract"]["covered_unit_count"],
            plan["source_unit_count"],
        )
        self.assertEqual(
            len(model_payload["evidence_digest"]["claim_coverage"]),
            plan["source_claim_count"],
        )
        split_outputs = [
            call.kwargs.get("output_json")
            for call in save_artifact.await_args_list
            if "_map_split_" in str(call.kwargs.get("artifact_key") or "")
        ]
        self.assertTrue(split_outputs)
        self.assertTrue(split_outputs[-1]["coverage_complete"])

    async def test_saved_split_reconciles_partial_parent_after_crash(
        self,
    ) -> None:
        payload = {
            "report_data": {
                "notes": [f"Содержательный вывод {index}" for index in range(4)]
            }
        }
        persisted: dict[str, Any] = {}
        parent_artifact_key: str | None = None
        successful_mapper = self._successful_mapper()

        def one_pack(
            units: list[dict[str, Any]],
            **_kwargs: Any,
        ) -> list[list[dict[str, Any]]]:
            return [units]

        async def partial_parent(
            *args: Any,
            **kwargs: Any,
        ) -> dict[str, Any]:
            nonlocal parent_artifact_key
            user_payload = kwargs["user_payload"]
            units = list(user_payload.get("source_units") or [])
            if len(units) > 2:
                parent_artifact_key = str(kwargs["artifact_key"])
                accepted_units = units[:2]
                accepted_ids = {
                    str(unit["source_unit_id"]) for unit in accepted_units
                }
                accepted_claims = [
                    claim
                    for claim in user_payload.get("source_claims") or []
                    if str(claim["source_unit_id"]) in accepted_ids
                ]
                return self._packet_for_units(
                    accepted_units,
                    source_claims=accepted_claims,
                )
            return await successful_mapper(*args, **kwargs)

        async def persist_then_crash(
            *_args: Any,
            **kwargs: Any,
        ) -> None:
            output = kwargs.get("output_json")
            artifact_key = str(kwargs.get("artifact_key") or "")
            if (
                "_map_split_" in artifact_key
                and isinstance(output, dict)
                and output.get("coverage_complete") is False
            ):
                persisted["artifact_key"] = artifact_key
                persisted["output"] = copy.deepcopy(output)
                raise RuntimeError("simulated crash after persisted split plan")

        with (
            patch(
                "app.services.analyzer._final_model_input_window",
                new_callable=AsyncMock,
                return_value=self._window(),
            ),
            patch(
                "app.services.analyzer._artifact_output",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "app.services.analyzer._pack_final_input_units",
                side_effect=one_pack,
            ),
            patch(
                "app.services.analyzer._structured_artifact",
                new_callable=AsyncMock,
                side_effect=partial_parent,
            ),
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
                side_effect=persist_then_crash,
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "simulated crash after persisted split plan",
            ):
                await _prepare_final_model_payload(
                    "run-id",
                    payload=payload,
                    system="author",
                    force_hierarchical=True,
                )

        self.assertIsNotNone(parent_artifact_key)
        self.assertIn("output", persisted)

        async def saved_split_output(
            _run_id: str,
            artifact_key: str,
            **_kwargs: Any,
        ) -> dict[str, Any] | None:
            if artifact_key == persisted["artifact_key"]:
                return copy.deepcopy(persisted["output"])
            return None

        with (
            patch(
                "app.services.analyzer._final_model_input_window",
                new_callable=AsyncMock,
                return_value=self._window(),
            ),
            patch(
                "app.services.analyzer._artifact_output",
                new_callable=AsyncMock,
                side_effect=saved_split_output,
            ),
            patch(
                "app.services.analyzer._pack_final_input_units",
                side_effect=one_pack,
            ),
            patch(
                "app.services.analyzer._structured_artifact",
                new_callable=AsyncMock,
                side_effect=successful_mapper,
            ) as structured_artifact,
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ) as save_artifact,
        ):
            _model_payload, plan = await _prepare_final_model_payload(
                "run-id",
                payload=payload,
                system="author",
                force_hierarchical=True,
            )

        self.assertEqual(plan["covered_claim_count"], plan["source_claim_count"])
        self.assertTrue(
            any(
                call.kwargs.get("artifact_key") == parent_artifact_key
                and call.kwargs.get("status") == "failed"
                and call.kwargs.get("preserve_existing_evidence") is True
                for call in save_artifact.await_args_list
            )
        )
        self.assertFalse(
            any(
                call.kwargs.get("artifact_key") == parent_artifact_key
                for call in structured_artifact.await_args_list
            )
        )

    async def test_incomplete_singleton_mapper_leaf_fails_once_without_loop(
        self,
    ) -> None:
        async def incomplete_mapper(
            *_args: Any,
            **_kwargs: Any,
        ) -> dict[str, Any]:
            return {
                "observations": [],
                "uncertainties": [],
                "report_focus": [],
                "unit_coverage": [],
                "claim_coverage": [],
            }

        with (
            patch(
                "app.services.analyzer._final_model_input_window",
                new_callable=AsyncMock,
                return_value=self._window(),
            ),
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ) as save_artifact,
            patch(
                "app.services.analyzer._artifact_output",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "app.services.analyzer._structured_artifact",
                new_callable=AsyncMock,
                side_effect=incomplete_mapper,
            ) as structured_artifact,
        ):
            with self.assertRaisesRegex(
                OpenRouterError,
                "incomplete dependent coverage",
            ):
                await _prepare_final_model_payload(
                    "run-id",
                    payload={"report_data": {"note": "Содержательный вывод"}},
                    system="author",
                    force_hierarchical=True,
                )

        self.assertEqual(structured_artifact.await_count, 1)
        self.assertFalse(
            any(
                "_map_split_" in str(call.kwargs.get("artifact_key") or "")
                for call in save_artifact.await_args_list
            )
        )

    async def test_mapper_unknown_identity_is_hard_failure_not_split(self) -> None:
        async def invented_identity(
            *_args: Any,
            **kwargs: Any,
        ) -> dict[str, Any]:
            user_payload = kwargs["user_payload"]
            packet = self._packet_for_units(
                user_payload["source_units"],
                source_claims=user_payload.get("source_claims"),
            )
            packet["observations"][0]["source_claim_ids"] = ["invented-claim"]
            return packet

        with (
            patch(
                "app.services.analyzer._final_model_input_window",
                new_callable=AsyncMock,
                return_value=self._window(),
            ),
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ) as save_artifact,
            patch(
                "app.services.analyzer._artifact_output",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "app.services.analyzer._structured_artifact",
                new_callable=AsyncMock,
                side_effect=invented_identity,
            ) as structured_artifact,
        ):
            with self.assertRaisesRegex(
                OpenRouterError,
                "invented source identities",
            ):
                await _prepare_final_model_payload(
                    "run-id",
                    payload={
                        "report_data": {
                            "notes": ["Первый вывод", "Второй вывод"]
                        }
                    },
                    system="author",
                    force_hierarchical=True,
                )

        self.assertEqual(structured_artifact.await_count, 1)
        self.assertFalse(
            any(
                "_map_split_" in str(call.kwargs.get("artifact_key") or "")
                for call in save_artifact.await_args_list
            )
        )

    async def test_mapper_pack_workload_is_bounded_by_claim_count(self) -> None:
        successful_mapper = self._successful_mapper()
        mapper_claim_counts: list[int] = []

        async def capture_workload(
            *args: Any,
            **kwargs: Any,
        ) -> dict[str, Any]:
            user_payload = kwargs["user_payload"]
            if "source_units" in user_payload:
                mapper_claim_counts.append(len(user_payload.get("source_claims") or []))
            return await successful_mapper(*args, **kwargs)

        with (
            patch(
                "app.services.analyzer._final_model_input_window",
                new_callable=AsyncMock,
                return_value=self._window(),
            ),
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ),
            patch(
                "app.services.analyzer._artifact_output",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "app.services.analyzer._structured_artifact",
                new_callable=AsyncMock,
                side_effect=capture_workload,
            ),
        ):
            _model_payload, plan = await _prepare_final_model_payload(
                "run-id",
                payload={
                    "report_data": {
                        "notes": [
                            f"Содержательный вывод {index}" for index in range(17)
                        ]
                    }
                },
                system="author",
                force_hierarchical=True,
            )

        self.assertGreater(len(mapper_claim_counts), 1)
        self.assertTrue(all(0 < count <= 8 for count in mapper_claim_counts))
        self.assertEqual(sum(mapper_claim_counts), plan["source_claim_count"])

    async def test_long_cyrillic_scalar_is_split_into_atomic_mapper_units(
        self,
    ) -> None:
        successful_mapper = self._successful_mapper()
        mapper_claim_counts: list[int] = []
        mapper_unit_claim_counts: list[list[int]] = []

        async def capture_workload(
            *args: Any,
            **kwargs: Any,
        ) -> dict[str, Any]:
            user_payload = kwargs["user_payload"]
            if "source_units" in user_payload:
                claims = list(user_payload.get("source_claims") or [])
                mapper_claim_counts.append(len(claims))
                counts = Counter(str(claim["source_unit_id"]) for claim in claims)
                mapper_unit_claim_counts.append(list(counts.values()))
            return await successful_mapper(*args, **kwargs)

        with (
            patch(
                "app.services.analyzer._final_model_input_window",
                new_callable=AsyncMock,
                return_value=self._window(),
            ),
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ),
            patch(
                "app.services.analyzer._artifact_output",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "app.services.analyzer._structured_artifact",
                new_callable=AsyncMock,
                side_effect=capture_workload,
            ),
        ):
            _model_payload, plan = await _prepare_final_model_payload(
                "run-id",
                payload={"report_data": {"note": "я" * 32_000}},
                system="author",
                force_hierarchical=True,
            )

        self.assertGreater(len(mapper_claim_counts), 1)
        self.assertTrue(all(0 < count <= 8 for count in mapper_claim_counts))
        self.assertTrue(
            all(count == 1 for pack in mapper_unit_claim_counts for count in pack)
        )
        self.assertEqual(sum(mapper_claim_counts), plan["source_claim_count"])

    @staticmethod
    def _packet_for_units(
        units: list[dict[str, Any]],
        *,
        source_claims: list[dict[str, Any]] | None = None,
        hide_values: bool = False,
    ) -> dict[str, Any]:
        claims = list(source_claims or [])
        if not claims:
            claims = [
                {
                    "claim_id": f"test-claim:{unit['source_unit_id']}",
                    "excerpt_sha256": "0" * 64,
                    "excerpt": str(unit.get("context_value") or ""),
                    "source_unit_id": str(unit["source_unit_id"]),
                    "source_path": str(unit["source_path"]),
                }
                for unit in units
            ]
        return {
            "observations": [
                {
                    "category": "context",
                    "statement": str(claim.get("excerpt") or "")[:40],
                    "source_paths": [str(claim["source_path"])],
                    "source_unit_ids": [str(claim["source_unit_id"])],
                    "source_claim_ids": [str(claim["claim_id"])],
                    "exact_values": (
                        [] if hide_values else [str(claim.get("excerpt") or "")[:40]]
                    ),
                    "evidence_excerpt": str(claim.get("excerpt") or "")[:40],
                    "importance": "supporting",
                }
                for claim in claims
            ],
            "uncertainties": [],
            "report_focus": [],
            "unit_coverage": [
                {
                    "source_unit_id": str(unit["source_unit_id"]),
                    "disposition": "supporting_context",
                    "rationale": "Единица отражена в observation.",
                }
                for unit in units
            ],
            "claim_coverage": [
                {
                    "claim_id": str(claim["claim_id"]),
                    "excerpt_sha256": str(claim["excerpt_sha256"]),
                    "disposition": "supporting_context",
                    "rationale": "Claim отражён в observation.",
                }
                for claim in claims
            ],
        }

    @staticmethod
    def _multi_claim_fixture() -> tuple[
        str,
        list[dict[str, Any]],
        list[dict[str, Any]],
        dict[str, Any],
    ]:
        source = (
            "FIRST-CLAIM-MARKER "
            + ("a" * 4_300)
            + " SECOND-CLAIM-MARKER "
            + ("b" * 4_300)
            + " TAIL-CLAIM-MARKER"
        )
        units, _manifest = _flatten_final_input_payload(
            {"report_data": {"narrative": source}},
            target_chars=20_000,
            context_overlap_chars=0,
        )
        claim_rows, claim_objects, _ids_by_unit, _ledger = _final_input_claim_ledger(
            units
        )
        if len(units) != 1 or len(claim_rows) < 2:
            raise AssertionError("Fixture must create one unit and many claims")
        return source, units, claim_rows, claim_objects

    def test_one_observation_cannot_claim_multiple_source_fragments(self) -> None:
        _source, units, claim_rows, claim_objects = self._multi_claim_fixture()
        packet = self._packet_for_units(
            units,
            source_claims=claim_rows,
        )
        packet["observations"][0]["source_claim_ids"] = [
            str(claim_rows[0]["claim_id"]),
            str(claim_rows[1]["claim_id"]),
        ]
        packet["observations"].pop(1)

        with self.assertRaisesRegex(
            OpenRouterError,
            "must bind exactly one source claim",
        ):
            _normalize_final_evidence_packet(
                packet,
                allowed_unit_paths={
                    str(unit["source_unit_id"]): str(unit["source_path"])
                    for unit in units
                },
                allowed_claims={str(claim["claim_id"]): claim for claim in claim_rows},
                claim_objects=claim_objects,
            )

    def test_atomic_claim_rows_keep_distinct_evidence_for_every_claim(
        self,
    ) -> None:
        source, units, claim_rows, claim_objects = self._multi_claim_fixture()
        packet = self._packet_for_units(
            units,
            source_claims=claim_rows,
        )
        normalized = _normalize_final_evidence_packet(
            packet,
            allowed_unit_paths={
                str(unit["source_unit_id"]): str(unit["source_path"]) for unit in units
            },
            allowed_claims={str(claim["claim_id"]): claim for claim in claim_rows},
            claim_objects=claim_objects,
        )

        observations_by_claim = {
            str(observation["source_claim_ids"][0]): observation
            for observation in normalized["observations"]
        }
        self.assertEqual(len(observations_by_claim), len(claim_rows))
        for claim in claim_rows:
            observation = observations_by_claim[str(claim["claim_id"])]
            self.assertIn(
                observation["evidence_excerpt"],
                str(claim["excerpt"]),
            )
            self.assertEqual(
                observation["source_unit_ids"],
                [claim["source_unit_id"]],
            )
            self.assertEqual(
                observation["source_paths"],
                [claim["source_path"]],
            )
        reconstructed = "".join(
            str(claim["excerpt"])
            for claim in sorted(
                claim_rows,
                key=lambda item: int(item["fragment_index"]),
            )
        )
        self.assertEqual(reconstructed, source)

    def test_mapper_allows_explanatory_metadata_without_weakening_lineage(
        self,
    ) -> None:
        digest = (
            "d674ba10bb17d62062c8bf2fc7271a08"
            "eaf493b36c7658b061886bcd0f1592b3"
        )
        units, _manifest = _flatten_final_input_payload(
            {"answer_corpus_manifest": {"critic_rows_sha256": digest}},
            target_chars=20_000,
            context_overlap_chars=0,
        )
        claim_rows, claim_objects, _ids_by_unit, _ledger = (
            _final_input_claim_ledger(units)
        )
        packet = self._packet_for_units(units, source_claims=claim_rows)
        packet["observations"][0]["statement"] = (
            "Поле critic_rows_sha256 содержит технический 64-символьный "
            "идентификатор; само по себе оно не измеряет AI visibility."
        )
        packet["observations"][0]["exact_values"] = []

        normalized = _normalize_final_evidence_packet(
            packet,
            allowed_unit_paths={
                str(unit["source_unit_id"]): str(unit["source_path"])
                for unit in units
            },
            allowed_claims={
                str(claim["claim_id"]): claim for claim in claim_rows
            },
            claim_objects=claim_objects,
        )

        self.assertEqual(
            normalized["observations"][0]["evidence_excerpt"],
            digest[:40],
        )
        self.assertEqual(
            normalized["observations"][0]["source_claim_ids"],
            [str(claim_rows[0]["claim_id"])],
        )

    def test_mapper_statement_may_quote_its_code_owned_source_path(self) -> None:
        expected_cells = [
            {"model": f"example/model-{index}"} for index in range(8)
        ]
        expected_cells[7]["model"] = "openai/gpt-chat-latest"
        units, _manifest = _flatten_final_input_payload(
            {"answer_corpus_manifest": {"expected_cells": expected_cells}},
            target_chars=20_000,
            context_overlap_chars=0,
        )
        claim_rows, claim_objects, _ids_by_unit, _ledger = (
            _final_input_claim_ledger(units)
        )
        claim = next(
            item
            for item in claim_rows
            if str(item["source_path"]).endswith("expected_cells/7/model")
        )
        unit = next(
            item
            for item in units
            if item["source_unit_id"] == claim["source_unit_id"]
        )
        packet = self._packet_for_units([unit], source_claims=[claim])
        packet["observations"][0]["statement"] = (
            "В expected_cells/7 поле model содержит "
            "«openai/gpt-chat-latest»."
        )

        normalized = _normalize_final_evidence_packet(
            packet,
            allowed_unit_paths={
                str(unit["source_unit_id"]): str(unit["source_path"])
            },
            allowed_claims={str(claim["claim_id"]): claim},
            claim_objects={
                str(claim["claim_id"]): claim_objects[str(claim["claim_id"])]
            },
        )

        self.assertEqual(
            normalized["observations"][0]["statement"],
            packet["observations"][0]["statement"],
        )

        fabricated = copy.deepcopy(packet)
        for fabricated_statement in (
            "Получено 7 ответов.",
            packet["observations"][0]["statement"] + " Видимость 777%.",
        ):
            with self.subTest(statement=fabricated_statement):
                fabricated["observations"][0]["statement"] = fabricated_statement
                sanitized = _normalize_final_evidence_packet(
                    fabricated,
                    allowed_unit_paths={
                        str(unit["source_unit_id"]): str(unit["source_path"])
                    },
                    allowed_claims={str(claim["claim_id"]): claim},
                    claim_objects={
                        str(claim["claim_id"]): claim_objects[
                            str(claim["claim_id"])
                        ]
                    },
                )
                self.assertEqual(
                    sanitized["observations"][0]["statement"],
                    "Исходный фрагмент сохранён без дополнительной интерпретации.",
                )
                self.assertEqual(
                    sanitized["observations"][0]["importance"],
                    "supporting",
                )
                self.assertEqual(
                    sanitized["observations"][0]["category"],
                    "context",
                )
                self.assertTrue(
                    {
                        "replace_ungrounded_observation_statement",
                        "mark_generic_observation_quarantined",
                    }.issubset(
                        {
                            item["operation"]
                            for item in sanitized[
                                "_aiv_final_input_grounding_filter"
                            ]["operations"]
                        }
                    )
                )

    def test_mapper_source_path_tokens_cannot_launder_assertions(self) -> None:
        units, _manifest = _flatten_final_input_payload(
            {
                "report_data": {
                    "available": False,
                    "openai": False,
                    "provider": {
                        "openai": {"gpt-99-invented": "нейтрально"}
                    },
                }
            },
            target_chars=20_000,
            context_overlap_chars=0,
        )
        claim_rows, claim_objects, _ids_by_unit, _ledger = (
            _final_input_claim_ledger(units)
        )
        claims_by_path = {
            str(claim["source_path"]): claim for claim in claim_rows
        }
        units_by_path = {str(unit["source_path"]): unit for unit in units}

        for source_path, fabricated_statement in (
            ("/report_data/available", "Статус available."),
            ("/report_data/openai", "Ответ дала OpenAI."),
            (
                "/report_data/provider/openai/gpt-99-invented",
                "Ответ сгенерирован в openai/gpt-99-invented.",
            ),
        ):
            with self.subTest(source_path=source_path):
                claim = claims_by_path[source_path]
                unit = units_by_path[source_path]
                packet = self._packet_for_units([unit], source_claims=[claim])
                packet["observations"][0]["statement"] = fabricated_statement
                sanitized = _normalize_final_evidence_packet(
                    packet,
                    allowed_unit_paths={
                        str(unit["source_unit_id"]): str(unit["source_path"])
                    },
                    allowed_claims={str(claim["claim_id"]): claim},
                    claim_objects={
                        str(claim["claim_id"]): claim_objects[
                            str(claim["claim_id"])
                        ]
                    },
                )
                self.assertEqual(
                    sanitized["observations"][0]["statement"],
                    "Исходный фрагмент сохранён без дополнительной интерпретации.",
                )
                audit = sanitized["_aiv_final_input_grounding_filter"]
                self.assertEqual(audit["replacement_count"], 1)
                self.assertNotIn(
                    fabricated_statement,
                    json.dumps(audit, ensure_ascii=False),
                )

    def test_mapper_sanitizes_ungrounded_assertion_grade_literals(
        self,
    ) -> None:
        digest = (
            "d674ba10bb17d62062c8bf2fc7271a08"
            "eaf493b36c7658b061886bcd0f1592b3"
        )
        units, _manifest = _flatten_final_input_payload(
            {"answer_corpus_manifest": {"critic_rows_sha256": digest}},
            target_chars=20_000,
            context_overlap_chars=0,
        )
        claim_rows, claim_objects, _ids_by_unit, _ledger = (
            _final_input_claim_ledger(units)
        )
        base = self._packet_for_units(units, source_claims=claim_rows)
        base["observations"][0]["exact_values"] = []

        for statement in (
            "Видимость достигла 99,9%.",
            "Бюджет составил 5000 ₽.",
            "Источник: https://invented.example/report.",
            "Статус проверки complete.",
            "Ответ дала модель openai/gpt-99-invented.",
        ):
            with self.subTest(statement=statement):
                packet = copy.deepcopy(base)
                packet["observations"][0]["statement"] = statement
                sanitized = _normalize_final_evidence_packet(
                    packet,
                    allowed_unit_paths={
                        str(unit["source_unit_id"]): str(unit["source_path"])
                        for unit in units
                    },
                    allowed_claims={
                        str(claim["claim_id"]): claim for claim in claim_rows
                    },
                    claim_objects=claim_objects,
                )
                self.assertEqual(
                    sanitized["observations"][0]["statement"],
                    "Исходный фрагмент сохранён без дополнительной интерпретации.",
                )
                audit = sanitized["_aiv_final_input_grounding_filter"]
                self.assertEqual(audit["replacement_count"], 1)
                self.assertNotIn(statement, json.dumps(audit, ensure_ascii=False))

    def test_typed_literal_grounding_uses_exact_semantic_classes(self) -> None:
        self.assertTrue(
            _final_root_tokens_are_grounded(
                (
                    "GEMINI показывает 66,7%; источник "
                    "https://example.com/audit. Пояснение про AI-доступность "
                    "может быть сформулировано свободно."
                ),
                source_texts=[
                    (
                        "Google Gemini: 66.7%. Источник "
                        "https://example.com/audit"
                    )
                ],
            )
        )
        self.assertTrue(
            _final_root_tokens_are_grounded(
                "Хэш описан как технический 64-символьный идентификатор.",
                source_texts=["d674ba10bb17d62062c8bf2fc7271a08"],
            )
        )
        self.assertTrue(
            _final_root_tokens_are_grounded(
                "Получено 1500 ответов; пара web/memory обработана.",
                source_texts=[
                    "Получено 1\u202f500 ответов; режим web/memory сравнивается."
                ],
            )
        )
        self.assertTrue(
            _final_root_tokens_are_grounded(
                (
                    "Сдвиг 5 p. p.; бюджет 100 USD; статус unavailable; "
                    "модель openai/gpt-5.6-terra."
                ),
                source_texts=[
                    (
                        "Сдвиг 5 percentage points; бюджет $100; "
                        "статус not available; модель openai/gpt-5.6-terra."
                    )
                ],
            )
        )
        self.assertTrue(
            _final_root_tokens_are_grounded(
                "Бюджет 1 000 ₽; служебный путь foo/bar учтён.",
                source_texts=["Бюджет RUB 1\u2009000; путь foo/bar сохранён."],
            )
        )
        self.assertTrue(
            _final_root_tokens_are_grounded(
                "Путь foo/bar обработан; состояние available.",
                source_texts=["Результат not only available; путь сохранён."],
            )
        )

        mismatches = (
            ("Доля 67%.", "Доля 7%."),
            ("Изменение 5 процентных пунктов.", "Изменение 5%."),
            ("Состояние unavailable.", "Состояние available."),
            ("Бюджет 100 $.", "Бюджет 100 ₽."),
            ("Бюджет $100.", "Бюджет ₽100."),
            ("Ответ дала ChatGPT.", "Ответ дала Gemini."),
            ("Google Search доступен.", "Ответ дала Gemini."),
            ("Источник rw.plus.", "Источник aiv.example."),
            (
                "Источник https://example.com/a.",
                "Источник https://example.com/b.",
            ),
            ("Измерено 42 ответа.", "Доля составляет 42%."),
            (
                "Модель google/gemini-3.6-flash-preview.",
                "Модель google/gemini-3.6-flash.",
            ),
            ("Источник example.com.", "Источник x.ai."),
            ("Проверка ещё не подтверждена.", "Состояние verified."),
            ("Проверка не доступна.", "Состояние available."),
            ("Проверка not failed.", "Состояние failed."),
            ("Проверка not currently available.", "Состояние available."),
            ("Проверка not unavailable.", "Состояние unavailable."),
            ("Проверка not necessarily available.", "Состояние available."),
            ("Проверка не всегда доступно.", "Состояние доступно."),
            ("Бюджет 100 USD.", "Бюджет $100 EUR."),
            ("Изменение -5%.", "Изменение +5%."),
            ("Бюджет -100 USD.", "Бюджет +100 USD."),
            ("Бюджет -$100.", "Бюджет $100."),
            ("Получено 1\n000 ответов.", "Получено 1000 ответов."),
            (
                "Модель openai/gpt-5.6-terra:online.",
                "Модель openai/gpt-5.6-terra:free.",
            ),
            ("Ответ сохранён.", "Получено 7 ответов."),
        )
        for source, statement in mismatches:
            with self.subTest(source=source, statement=statement):
                self.assertFalse(
                    _final_root_tokens_are_grounded(
                        statement,
                        source_texts=[source],
                    )
                )

    def test_mapper_filters_auxiliary_literals_without_losing_hard_evidence(
        self,
    ) -> None:
        units, _manifest = _flatten_final_input_payload(
            {"report_data": {"note": "Доля 7%."}},
            target_chars=20_000,
            context_overlap_chars=0,
        )
        claim_rows, claim_objects, _ids_by_unit, _ledger = (
            _final_input_claim_ledger(units)
        )
        allowed_paths = {
            str(unit["source_unit_id"]): str(unit["source_path"])
            for unit in units
        }
        allowed_claims = {
            str(claim["claim_id"]): claim for claim in claim_rows
        }

        cases = {
            "statement": {
                "mutate": lambda packet: packet["observations"][0].update(
                    statement="Gemini показывает 88,8%."
                ),
                "operation": "replace_ungrounded_observation_statement",
                "assert_sanitized": lambda packet: self.assertEqual(
                    packet["observations"][0]["statement"],
                    "Исходный фрагмент сохранён без дополнительной интерпретации.",
                ),
            },
            "exact_values": {
                "mutate": lambda packet: packet["observations"][0].update(
                    exact_values=["99,9%"]
                ),
                "operation": "drop_ungrounded_exact_value",
                "assert_sanitized": lambda packet: self.assertEqual(
                    packet["observations"][0]["exact_values"],
                    [],
                ),
            },
            "uncertainties": {
                "mutate": lambda packet: packet.update(
                    uncertainties=[
                        "Видимость 99,9% по модели "
                        "openai/gpt-99-invented."
                    ]
                ),
                "operation": "drop_ungrounded_uncertainty",
                "assert_sanitized": lambda packet: self.assertEqual(
                    packet["uncertainties"],
                    [],
                ),
            },
            "report_focus": {
                "mutate": lambda packet: packet.update(
                    report_focus=["Бюджет $5000, статус complete."]
                ),
                "operation": "drop_ungrounded_report_focus",
                "assert_sanitized": lambda packet: self.assertEqual(
                    packet["report_focus"],
                    [],
                ),
            },
            "unit_coverage": {
                "mutate": lambda packet: packet["unit_coverage"][0].update(
                    rationale="Gemini показывает 88,8%."
                ),
                "operation": "replace_ungrounded_unit_coverage_rationale",
                "assert_sanitized": lambda packet: self.assertEqual(
                    packet["unit_coverage"][0]["rationale"],
                    "Исходная единица учтена в покрытии.",
                ),
            },
            "claim_coverage": {
                "mutate": lambda packet: packet["claim_coverage"][0].update(
                    rationale="Источник https://invented.example/report."
                ),
                "operation": "replace_ungrounded_claim_coverage_rationale",
                "assert_sanitized": lambda packet: self.assertEqual(
                    packet["claim_coverage"][0]["rationale"],
                    "Исходный фрагмент учтён в покрытии.",
                ),
            },
        }
        for field, case in cases.items():
            with self.subTest(field=field):
                packet = self._packet_for_units(
                    units,
                    source_claims=claim_rows,
                )
                case["mutate"](packet)
                normalized = _normalize_final_evidence_packet(
                    packet,
                    allowed_unit_paths=allowed_paths,
                    allowed_claims=allowed_claims,
                    claim_objects=claim_objects,
                )
                case["assert_sanitized"](normalized)
                audit = normalized["_aiv_final_input_grounding_filter"]
                self.assertEqual(audit["quality_state"], "degraded")
                expected_operations = {case["operation"]}
                if field == "statement":
                    expected_operations.add(
                        "mark_generic_observation_quarantined"
                    )
                self.assertEqual(
                    {
                        item["operation"] for item in audit["operations"]
                    },
                    expected_operations,
                )
                self.assertNotIn(
                    "99,9%",
                    json.dumps(audit, ensure_ascii=False),
                )

    def test_mapper_keeps_structural_failures_hard_and_statement_fail_soft(
        self,
    ) -> None:
        units, _manifest = _flatten_final_input_payload(
            {"report_data": {"note": "Доля 7%."}},
            target_chars=20_000,
            context_overlap_chars=0,
        )
        claim_rows, claim_objects, _ids_by_unit, _ledger = (
            _final_input_claim_ledger(units)
        )
        allowed_paths = {
            str(unit["source_unit_id"]): str(unit["source_path"])
            for unit in units
        }
        allowed_claims = {
            str(claim["claim_id"]): claim for claim in claim_rows
        }
        packet = self._packet_for_units(units, source_claims=claim_rows)
        packet["observations"][0]["exact_values"] = [7]
        with self.assertRaisesRegex(OpenRouterError, "invalid exact values"):
            _normalize_final_evidence_packet(
                packet,
                allowed_unit_paths=allowed_paths,
                allowed_claims=allowed_claims,
                claim_objects=claim_objects,
            )

        packet = self._packet_for_units(units, source_claims=claim_rows)
        packet["observations"][0]["statement"] = "Доля 99,9%."
        packet["unit_coverage"][0]["disposition"] = "material_observation"
        packet["claim_coverage"][0]["disposition"] = "material_observation"
        digest = str(packet["claim_coverage"][0]["excerpt_sha256"])
        packet["claim_coverage"][0]["excerpt_sha256"] = (
            digest[:18] + digest[19:]
        )
        sanitized = _normalize_final_evidence_packet(
            packet,
            allowed_unit_paths=allowed_paths,
            allowed_claims=allowed_claims,
            claim_objects=claim_objects,
        )
        self.assertEqual(
            sanitized["observations"][0]["statement"],
            "Исходный фрагмент сохранён без дополнительной интерпретации.",
        )
        self.assertEqual(
            sanitized["unit_coverage"][0]["disposition"],
            "supporting_context",
        )
        self.assertEqual(
            sanitized["claim_coverage"][0]["disposition"],
            "supporting_context",
        )
        operations = {
            item["operation"]
            for item in sanitized["_aiv_final_input_grounding_filter"][
                "operations"
            ]
        }
        self.assertTrue(
            {
                "replace_ungrounded_observation_statement",
                "replace_quarantined_unit_coverage_disposition",
                "replace_quarantined_claim_coverage_disposition",
                "replace_invalid_claim_coverage_digest",
            }.issubset(operations)
        )
        self.assertEqual(
            _normalize_final_evidence_packet(
                sanitized,
                allowed_unit_paths=allowed_paths,
                allowed_claims=allowed_claims,
                claim_objects=claim_objects,
            ),
            sanitized,
        )
        root_entries = _final_root_semantic_entries(
            {
                "evidence_digest": sanitized,
                "deterministic_passthrough": {"values": []},
                "long_input_contract": {
                    "source_claim_count": len(claim_rows)
                },
            },
            source_claim_rows=claim_rows,
        )
        self.assertEqual(
            [
                entry["value"]
                for entry in root_entries
                if entry["kind"] == "exact_source_claim"
            ],
            ["Доля 7%."],
        )
        self.assertNotIn(
            "99,9%",
            json.dumps(root_entries, ensure_ascii=False),
        )

    def test_mapper_rebinds_redundant_claim_digest_to_code_owned_ledger(
        self,
    ) -> None:
        units, _manifest = _flatten_final_input_payload(
            {"report_data": {"mode": "web"}},
            target_chars=20_000,
            context_overlap_chars=0,
        )
        claim_rows, claim_objects, _ids_by_unit, _ledger = (
            _final_input_claim_ledger(units)
        )
        claim = claim_rows[0]
        claim_id = str(claim["claim_id"])
        expected_digest = str(claim["excerpt_sha256"])
        packet = self._packet_for_units(units, source_claims=claim_rows)
        packet["observations"][0].update(
            statement="web",
            exact_values=["web"],
            evidence_excerpt="web",
        )
        packet["unit_coverage"][0]["rationale"] = "web"
        packet["claim_coverage"][0]["rationale"] = "web"
        packet["claim_coverage"][0]["excerpt_sha256"] = (
            expected_digest[:18] + expected_digest[19:]
        )

        normalized = _normalize_final_evidence_packet(
            packet,
            allowed_unit_paths={
                str(unit["source_unit_id"]): str(unit["source_path"])
                for unit in units
            },
            allowed_claims={claim_id: claim},
            claim_objects={claim_id: claim_objects[claim_id]},
        )

        self.assertEqual(
            normalized["claim_coverage"][0]["excerpt_sha256"],
            expected_digest,
        )
        audit = normalized["_aiv_final_input_grounding_filter"]
        self.assertEqual(audit["quality_state"], "degraded")
        self.assertEqual(audit["operation_count"], 1)
        self.assertEqual(
            audit["operations"][0]["operation"],
            "replace_invalid_claim_coverage_digest",
        )
        self.assertNotIn(
            expected_digest,
            json.dumps(audit, ensure_ascii=False),
        )
        self.assertEqual(
            _normalize_final_evidence_packet(
                normalized,
                allowed_unit_paths={
                    str(unit["source_unit_id"]): str(unit["source_path"])
                    for unit in units
                },
                allowed_claims={claim_id: claim},
                claim_objects={claim_id: claim_objects[claim_id]},
            ),
            normalized,
        )

        corrupt_claim = copy.deepcopy(claim)
        corrupt_claim["excerpt_sha256"] = "f" * 64
        with self.assertRaisesRegex(
            OpenRouterError,
            "corrupt code-owned claim digest",
        ):
            _normalize_final_evidence_packet(
                packet,
                allowed_unit_paths={
                    str(unit["source_unit_id"]): str(unit["source_path"])
                    for unit in units
                },
                allowed_claims={claim_id: corrupt_claim},
                claim_objects={claim_id: claim_objects[claim_id]},
            )

    def test_generic_mapper_ack_cannot_retain_material_priority(self) -> None:
        units, _manifest = _flatten_final_input_payload(
            {"report_data": {"mode": "web"}},
            target_chars=20_000,
            context_overlap_chars=0,
        )
        claim_rows, claim_objects, _ids_by_unit, _ledger = (
            _final_input_claim_ledger(units)
        )
        claim = claim_rows[0]
        claim_id = str(claim["claim_id"])
        allowed_paths = {
            str(unit["source_unit_id"]): str(unit["source_path"])
            for unit in units
        }
        for statement in (
            "Исходная единица учтена в покрытии.",
            "Исходный фрагмент учтён в покрытии.",
            "Фрагмент сохранён без дополнительной интерпретации.",
            "Claim accounted in coverage.",
            "Данные успешно сохранены.",
            "Исходный фрагмент обработан корректно.",
            "Путь /report_data/mode учтён в покрытии.",
            "В /report_data/mode данные обработаны корректно.",
            "Source path /report_data/mode recorded correctly.",
            "web обработан корректно.",
            "web успешно сохранён.",
            "web processed successfully.",
            "web обработан блестяще.",
            "web processed rapidly.",
            ".",
            "e",
            "we",
            "—",
        ):
            with self.subTest(statement=statement):
                packet = self._packet_for_units(
                    units,
                    source_claims=claim_rows,
                )
                packet["observations"][0].update(
                    statement=statement,
                    category="visibility",
                    importance="critical",
                    exact_values=["web"],
                    evidence_excerpt="web",
                )
                packet["unit_coverage"][0].update(
                    disposition="material_observation",
                    rationale="web",
                )
                packet["claim_coverage"][0].update(
                    disposition="material_observation",
                    rationale="web",
                )

                normalized = _normalize_final_evidence_packet(
                    packet,
                    allowed_unit_paths=allowed_paths,
                    allowed_claims={claim_id: claim},
                    claim_objects={claim_id: claim_objects[claim_id]},
                )

                self.assertEqual(
                    normalized["observations"][0]["category"],
                    "context",
                )
                self.assertEqual(
                    normalized["observations"][0]["importance"],
                    "supporting",
                )
                self.assertEqual(
                    normalized["unit_coverage"][0]["disposition"],
                    "supporting_context",
                )
                self.assertEqual(
                    normalized["claim_coverage"][0]["disposition"],
                    "supporting_context",
                )
                self.assertTrue(
                    {
                        "mark_generic_observation_quarantined",
                        "replace_quarantined_observation_category",
                        "replace_quarantined_observation_importance",
                        "replace_quarantined_unit_coverage_disposition",
                        "replace_quarantined_claim_coverage_disposition",
                    }.issubset(
                        {
                            item["operation"]
                            for item in normalized[
                                "_aiv_final_input_grounding_filter"
                            ]["operations"]
                        }
                    )
                )
                self.assertEqual(
                    _normalize_final_evidence_packet(
                        normalized,
                        allowed_unit_paths=allowed_paths,
                        allowed_claims={claim_id: claim},
                        claim_objects={claim_id: claim_objects[claim_id]},
                    ),
                    normalized,
                )
        material_packet = self._packet_for_units(
            units,
            source_claims=claim_rows,
        )
        material_packet["observations"][0].update(
            statement="В /report_data/mode значение web.",
            category="visibility",
            importance="critical",
            exact_values=["web"],
            evidence_excerpt="web",
        )
        material_packet["unit_coverage"][0].update(
            disposition="material_observation",
            rationale="web",
        )
        material_packet["claim_coverage"][0].update(
            disposition="material_observation",
            rationale="web",
        )
        material = _normalize_final_evidence_packet(
            material_packet,
            allowed_unit_paths=allowed_paths,
            allowed_claims={claim_id: claim},
            claim_objects={claim_id: claim_objects[claim_id]},
        )
        self.assertEqual(material["observations"][0]["category"], "visibility")
        self.assertEqual(material["observations"][0]["importance"], "critical")
        self.assertEqual(
            material["unit_coverage"][0]["disposition"],
            "material_observation",
        )
        self.assertEqual(
            material["claim_coverage"][0]["disposition"],
            "material_observation",
        )
        self.assertNotIn("_aiv_final_input_grounding_filter", material)
        root_entries = _final_root_semantic_entries(
            {
                "evidence_digest": normalized,
                "deterministic_passthrough": {"values": []},
                "long_input_contract": {"source_claim_count": 1},
            },
            source_claim_rows=claim_rows,
        )
        self.assertEqual(
            [
                entry["value"]
                for entry in root_entries
                if entry["kind"] == "exact_source_claim"
            ],
            ["web"],
        )

    def test_mapper_auxiliary_filter_audit_is_idempotent_through_union(
        self,
    ) -> None:
        units, _manifest = _flatten_final_input_payload(
            {"report_data": {"note": "Доля 7%."}},
            target_chars=20_000,
            context_overlap_chars=0,
        )
        claim_rows, claim_objects, _ids_by_unit, _ledger = (
            _final_input_claim_ledger(units)
        )
        allowed_paths = {
            str(unit["source_unit_id"]): str(unit["source_path"])
            for unit in units
        }
        allowed_claims = {
            str(claim["claim_id"]): claim for claim in claim_rows
        }
        packet = self._packet_for_units(units, source_claims=claim_rows)
        packet["unit_coverage"][0]["rationale"] = "Статус complete."
        normalized = _normalize_final_evidence_packet(
            packet,
            allowed_unit_paths=allowed_paths,
            allowed_claims=allowed_claims,
            claim_objects=claim_objects,
        )
        audit = normalized["_aiv_final_input_grounding_filter"]

        renormalized = _normalize_final_evidence_packet(
            normalized,
            allowed_unit_paths=allowed_paths,
            allowed_claims=allowed_claims,
            claim_objects=claim_objects,
        )
        self.assertEqual(renormalized, normalized)
        self.assertEqual(
            renormalized["_aiv_final_input_grounding_filter"],
            audit,
        )

        union = _preserve_final_evidence_reduction(
            {
                "observations": [],
                "uncertainties": [],
                "report_focus": [],
                "unit_coverage": [],
                "claim_coverage": [],
            },
            [normalized],
        )
        union = _normalize_final_evidence_packet(
            union,
            allowed_unit_paths=allowed_paths,
            allowed_claims=allowed_claims,
            claim_objects=claim_objects,
        )
        self.assertEqual(
            union["_aiv_final_input_grounding_filter"],
            audit,
        )

        tampered = copy.deepcopy(normalized)
        tampered["_aiv_final_input_grounding_filter"]["operation_count"] += 1
        with self.assertRaisesRegex(OpenRouterError, "filter audit is invalid"):
            _normalize_final_evidence_packet(
                tampered,
                allowed_unit_paths=allowed_paths,
                allowed_claims=allowed_claims,
                claim_objects=claim_objects,
            )

    def test_bounded_root_exact_excerpt_cannot_strip_literal_context(self) -> None:
        cases = (
            ("Status: not available.", "available"),
            ("Бюджет $100.", "100"),
            ("Доля 5%.", "5"),
            ("Модель openai/gpt-5.6-terra.", "gpt-5.6"),
        )
        for source_text, excerpt in cases:
            with self.subTest(source_text=source_text, excerpt=excerpt):
                packet = {
                    "observations": [
                        {
                            "category": "context",
                            "statement": excerpt,
                            "source_node_ids": ["node-1"],
                            "exact_values": [],
                            "fact_binding_ids": [],
                            "evidence_excerpts": [
                                {
                                    "source_node_id": "node-1",
                                    "excerpt": excerpt,
                                }
                            ],
                            "importance": "supporting",
                        }
                    ],
                    "uncertainties": [],
                    "report_focus": [],
                    "node_coverage": [
                        {
                            "source_node_id": "node-1",
                            "disposition": "supporting_context",
                            "rationale": "Источник учтён.",
                        }
                    ],
                }
                with self.assertRaisesRegex(
                    OpenRouterError,
                    "invented an exact literal or state",
                ):
                    _normalize_final_root_summary_packet(
                        packet,
                        allowed_node_text={"node-1": source_text},
                        allowed_node_fact_bindings={"node-1": []},
                    )

    def test_bounded_root_replaces_ungrounded_coverage_rationale(self) -> None:
        packet = {
            "observations": [
                {
                    "category": "visibility",
                    "statement": "Доля 7%.",
                    "source_node_ids": ["node-1"],
                    "exact_values": ["7%"],
                    "fact_binding_ids": [],
                    "evidence_excerpts": [
                        {"source_node_id": "node-1", "excerpt": "Доля 7%."}
                    ],
                    "importance": "important",
                }
            ],
            "uncertainties": [],
            "report_focus": [],
            "node_coverage": [
                {
                    "source_node_id": "node-1",
                    "disposition": "material_observation",
                    "rationale": (
                        "Gemini показывает 99,9%; источник "
                        "https://invented.example."
                    ),
                }
            ],
        }
        normalized = _normalize_final_root_summary_packet(
            packet,
            allowed_node_text={"node-1": "Доля 7%."},
            allowed_node_fact_bindings={"node-1": []},
        )
        self.assertEqual(
            normalized["node_coverage"][0]["rationale"],
            "Дочерний узел учтён в покрытии.",
        )
        audit = normalized["_aiv_final_input_grounding_filter"]
        self.assertEqual(audit["operation_count"], 1)
        self.assertEqual(audit["replacement_count"], 1)
        self.assertEqual(
            audit["operations"][0]["operation"],
            "replace_ungrounded_node_coverage_rationale",
        )
        self.assertEqual(
            _normalize_final_root_summary_packet(
                normalized,
                allowed_node_text={"node-1": "Доля 7%."},
                allowed_node_fact_bindings={"node-1": []},
            ),
            normalized,
        )

    def test_bounded_root_drops_only_ungrounded_auxiliary_items(self) -> None:
        packet = {
            "observations": [
                {
                    "category": "visibility",
                    "statement": "Доля 7%.",
                    "source_node_ids": ["node-1"],
                    "exact_values": ["7%", "99,9%"],
                    "fact_binding_ids": [],
                    "evidence_excerpts": [
                        {"source_node_id": "node-1", "excerpt": "Доля 7%."}
                    ],
                    "importance": "important",
                }
            ],
            "uncertainties": [
                {"text": "Доля 7%.", "source_node_ids": ["node-1"]},
                {"text": "Статус complete.", "source_node_ids": ["node-1"]},
            ],
            "report_focus": [
                {"text": "Бюджет $5000.", "source_node_ids": ["node-1"]}
            ],
            "node_coverage": [
                {
                    "source_node_id": "node-1",
                    "disposition": "material_observation",
                    "rationale": "Доля 7%.",
                }
            ],
        }
        normalized = _normalize_final_root_summary_packet(
            packet,
            allowed_node_text={"node-1": "Доля 7%."},
            allowed_node_fact_bindings={"node-1": []},
        )
        self.assertEqual(normalized["observations"][0]["exact_values"], ["7%"])
        self.assertEqual(
            normalized["uncertainties"],
            [{"text": "Доля 7%.", "source_node_ids": ["node-1"]}],
        )
        self.assertEqual(normalized["report_focus"], [])
        audit = normalized["_aiv_final_input_grounding_filter"]
        self.assertEqual(audit["operation_count"], 3)
        self.assertEqual(audit["drop_count"], 3)
        self.assertEqual(
            {
                item["operation"] for item in audit["operations"]
            },
            {
                "drop_ungrounded_exact_value",
                "drop_ungrounded_uncertainty",
                "drop_ungrounded_report_focus",
            },
        )

    def test_reducer_cannot_replace_atomic_claim_rows_with_rephrases(
        self,
    ) -> None:
        _source, units, claim_rows, claim_objects = self._multi_claim_fixture()
        canonical = _normalize_final_evidence_packet(
            self._packet_for_units(units, source_claims=claim_rows),
            allowed_unit_paths={
                str(unit["source_unit_id"]): str(unit["source_path"]) for unit in units
            },
            allowed_claims={str(claim["claim_id"]): claim for claim in claim_rows},
            claim_objects=claim_objects,
        )
        rewritten = copy.deepcopy(canonical)
        for index, observation in enumerate(rewritten["observations"]):
            observation["statement"] = f"Reducer rewrite {index}"
            observation["evidence_excerpt"] = "shortened"

        preserved = _preserve_final_evidence_reduction(
            rewritten,
            [canonical],
        )

        self.assertEqual(
            preserved["observations"],
            canonical["observations"],
        )
        self.assertEqual(
            [
                observation["source_claim_ids"][0]
                for observation in preserved["observations"]
            ],
            [str(claim["claim_id"]) for claim in claim_rows],
        )

    @classmethod
    def _successful_mapper(
        cls,
        *,
        hide_values: bool = False,
    ) -> Any:
        async def mapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
            user_payload = kwargs["user_payload"]
            if "source_units" in user_payload:
                return cls._packet_for_units(
                    user_payload["source_units"],
                    source_claims=user_payload.get("source_claims"),
                    hide_values=hide_values,
                )
            if "source_nodes" in user_payload:
                source_nodes = user_payload["source_nodes"]

                def semantic_text(node: dict[str, Any]) -> str:
                    if isinstance(node.get("summary"), dict):
                        return json.dumps(
                            node["summary"],
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    context = str(node.get("context_value") or "")
                    start = int(node.get("core_start_in_context") or 0)
                    end = int(node.get("core_end_in_context") or len(context))
                    return context[start:end]

                def grounded_excerpt(node: dict[str, Any]) -> str:
                    value = semantic_text(node)
                    return value[:240] if value else "Пустое значение."

                def fact_table_values(
                    node: dict[str, Any],
                    column: str,
                ) -> list[str]:
                    table = node.get("mandatory_fact_table") or {}
                    columns = table.get("columns") or []
                    rows = table.get("rows") or []
                    if column not in columns:
                        return []
                    index = columns.index(column)
                    return [
                        str(row[index])
                        for row in rows
                        if isinstance(row, list)
                        and len(row) > index
                        and row[index] not in {None, ""}
                    ]

                def grounded_statement(node: dict[str, Any]) -> str:
                    excerpt = grounded_excerpt(node)
                    lexemes = fact_table_values(node, "lexeme")
                    if not lexemes:
                        return excerpt
                    return f"{excerpt} Факты: {'; '.join(lexemes)}."

                return {
                    "observations": [
                        {
                            "category": "context",
                            "statement": grounded_statement(node),
                            "source_node_ids": [str(node["source_node_id"])],
                            "exact_values": [],
                            "fact_binding_ids": fact_table_values(node, "ref"),
                            "evidence_excerpts": [
                                {
                                    "source_node_id": str(node["source_node_id"]),
                                    "excerpt": grounded_excerpt(node),
                                }
                            ],
                            "importance": "supporting",
                        }
                        for node in source_nodes
                    ],
                    "uncertainties": [],
                    "report_focus": [],
                    "node_coverage": [
                        {
                            "source_node_id": str(node["source_node_id"]),
                            "disposition": "supporting_context",
                            "rationale": "Child node отражён в observation.",
                        }
                        for node in source_nodes
                    ],
                }
            packets = user_payload["evidence_packets"]
            return {
                "observations": [
                    copy.deepcopy(observation)
                    for packet in packets
                    for observation in packet["observations"]
                ],
                "uncertainties": [],
                "report_focus": [],
                "unit_coverage": [
                    copy.deepcopy(item)
                    for packet in packets
                    for item in packet["unit_coverage"]
                ],
                "claim_coverage": [
                    copy.deepcopy(item)
                    for packet in packets
                    for item in packet["claim_coverage"]
                ],
            }

        return mapper

    def test_mapper_cannot_swap_unit_and_source_path_provenance(self) -> None:
        packet = {
            "observations": [
                {
                    "category": "context",
                    "statement": "Переставленная ссылка A.",
                    "source_paths": ["/b"],
                    "source_unit_ids": ["unit-a"],
                    "exact_values": [],
                    "evidence_excerpt": "A",
                    "importance": "supporting",
                },
                {
                    "category": "context",
                    "statement": "Переставленная ссылка B.",
                    "source_paths": ["/a"],
                    "source_unit_ids": ["unit-b"],
                    "exact_values": [],
                    "evidence_excerpt": "B",
                    "importance": "supporting",
                },
            ],
            "uncertainties": [],
            "report_focus": [],
            "unit_coverage": [
                {
                    "source_unit_id": "unit-a",
                    "disposition": "supporting_context",
                    "rationale": "A учтена.",
                },
                {
                    "source_unit_id": "unit-b",
                    "disposition": "supporting_context",
                    "rationale": "B учтена.",
                },
            ],
        }

        with self.assertRaisesRegex(
            OpenRouterError,
            "mismatches its code-owned source unit paths",
        ):
            _normalize_final_evidence_packet(
                packet,
                allowed_unit_paths={"unit-a": "/a", "unit-b": "/b"},
            )

    async def test_model_window_never_exceeds_residual_context(self) -> None:
        envelope = {
            "version": "test",
            "resolution": "test",
            "context_length": 16_000,
            "max_completion_tokens": 16_000,
        }
        with patch(
            "app.services.analyzer.model_output_envelope",
            new_callable=AsyncMock,
            return_value=envelope,
        ):
            with self.assertRaisesRegex(
                OpenRouterError,
                "leaves no safe input window",
            ):
                await _final_model_input_window(model=ANALYSIS_MODEL)

        positive_envelope = {
            **envelope,
            "context_length": 36_000,
            "max_completion_tokens": 20_000,
        }
        with patch(
            "app.services.analyzer.model_output_envelope",
            new_callable=AsyncMock,
            return_value=positive_envelope,
        ):
            window = await _final_model_input_window(model=ANALYSIS_MODEL)

        self.assertEqual(
            window["input_token_window"],
            36_000 - 20_000,
        )
        self.assertEqual(
            window["input_utf8_window"],
            window["input_token_window"],
        )
        self.assertLessEqual(
            window["input_token_window"] + positive_envelope["max_completion_tokens"],
            positive_envelope["context_length"],
        )

    async def test_all_analyzer_planners_share_one_byte_per_token_contract(
        self,
    ) -> None:
        envelope = {
            "version": "test",
            "resolution": "test",
            "context_length": 50_000,
            "max_completion_tokens": 10_000,
        }
        expected = 50_000 - 10_000
        with patch(
            "app.services.analyzer.model_output_envelope",
            new_callable=AsyncMock,
            return_value=envelope,
        ):
            analyzer_window = await _analyzer_model_input_window(
                PROCESSING_MODEL,
                system="Системная инструкция",
            )
            market_window = await _market_research_model_window(
                system="Исследовательская инструкция",
            )
            final_window = await _final_model_input_window(
                model=ANALYSIS_MODEL,
            )

        for window in (analyzer_window, market_window, final_window):
            self.assertEqual(window["input_token_window"], expected)
            self.assertEqual(window["input_utf8_window"], expected)
            self.assertEqual(
                window["utf8_preflight_contract"],
                "exact_serialized_request_utf8_bytes<=residual_input_tokens",
            )

    async def test_oversized_payload_has_complete_unit_coverage(self) -> None:
        with (
            patch(
                "app.services.analyzer._final_model_input_window",
                new_callable=AsyncMock,
                return_value=self._window(),
            ),
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ),
            patch(
                "app.services.analyzer._structured_artifact",
                new_callable=AsyncMock,
                side_effect=self._successful_mapper(),
            ),
        ):
            model_payload, preflight = await _prepare_final_model_payload(
                "run-id",
                payload=self._oversized_payload(),
                system="author",
            )

        contract = model_payload["long_input_contract"]
        self.assertEqual(preflight["mode"], "hierarchical_evidence_tree")
        self.assertTrue(contract["coverage_complete"])
        self.assertEqual(
            contract["covered_unit_count"],
            contract["source_unit_count"],
        )
        self.assertEqual(
            len(model_payload["evidence_digest"]["unit_coverage"]),
            contract["source_unit_count"],
        )

    async def test_long_narrative_claims_cover_tail_exactly_once(self) -> None:
        tail_marker = "TAIL-CLAIM-MUST-SURVIVE-7c2d3f"
        narrative = "x" * 14_000 + tail_marker
        seen_source_claims: list[dict[str, Any]] = []

        async def capture_mapper(
            *_args: Any,
            **kwargs: Any,
        ) -> dict[str, Any]:
            source_claims = kwargs["user_payload"].get("source_claims")
            self.assertIsInstance(source_claims, list)
            seen_source_claims.extend(copy.deepcopy(source_claims))
            return self._packet_for_units(
                kwargs["user_payload"]["source_units"],
                source_claims=source_claims,
            )

        with (
            patch(
                "app.services.analyzer._final_model_input_window",
                new_callable=AsyncMock,
                return_value=self._window(),
            ),
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ),
            patch(
                "app.services.analyzer._structured_artifact",
                new_callable=AsyncMock,
                side_effect=capture_mapper,
            ),
        ):
            model_payload, plan = await _prepare_final_model_payload(
                "run-id",
                payload={"report_data": {"long_narrative": narrative}},
                system="author",
                force_hierarchical=True,
            )

        self.assertGreater(plan["source_claim_count"], 1)
        self.assertEqual(
            plan["source_claim_count"],
            plan["source_unit_count"],
        )
        self.assertEqual(
            plan["covered_claim_count"],
            plan["source_claim_count"],
        )
        self.assertEqual(
            len(model_payload["evidence_digest"]["claim_coverage"]),
            plan["source_claim_count"],
        )
        self.assertEqual(
            len(
                {
                    item["claim_id"]
                    for item in model_payload["evidence_digest"]["claim_coverage"]
                }
            ),
            plan["source_claim_count"],
        )
        reconstructed = "".join(
            str(claim["excerpt"])
            for claim in sorted(
                seen_source_claims,
                key=lambda claim: (
                    int(claim["source_core_start_char"]),
                    int(claim["fragment_index"]),
                ),
            )
        )
        self.assertEqual(reconstructed, narrative)
        self.assertTrue(reconstructed.endswith(tail_marker))

    async def test_missing_claim_fails_closed(self) -> None:
        narrative = "x" * 14_000 + "TAIL-CLAIM-FAIL-CLOSED"

        async def broken_mapper(
            *_args: Any,
            **kwargs: Any,
        ) -> dict[str, Any]:
            source_claims = kwargs["user_payload"]["source_claims"]
            self.assertGreaterEqual(len(source_claims), 1)
            packet = self._packet_for_units(
                kwargs["user_payload"]["source_units"],
                source_claims=source_claims,
            )
            packet["observations"].pop()
            packet["claim_coverage"].pop()
            return packet

        with (
            patch(
                "app.services.analyzer._final_model_input_window",
                new_callable=AsyncMock,
                return_value=self._window(),
            ),
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ),
            patch(
                "app.services.analyzer._structured_artifact",
                new_callable=AsyncMock,
                side_effect=broken_mapper,
            ),
        ):
            with self.assertRaisesRegex(
                OpenRouterError,
                "incomplete dependent coverage",
            ):
                await _prepare_final_model_payload(
                    "run-id",
                    payload={
                        "report_data": {
                            "long_narrative": narrative,
                        }
                    },
                    system="author",
                    force_hierarchical=True,
                )

    async def test_mapper_digest_echo_typo_is_repaired_in_hierarchical_path(
        self,
    ) -> None:
        narrative = "x" * 14_000 + "TAIL-CLAIM-REBOUND"
        successful_mapper = self._successful_mapper()

        async def digest_typo_mapper(
            *_args: Any,
            **kwargs: Any,
        ) -> dict[str, Any]:
            user_payload = kwargs["user_payload"]
            if "source_units" not in user_payload:
                return await successful_mapper(*_args, **kwargs)
            source_claims = user_payload["source_claims"]
            packet = self._packet_for_units(
                user_payload["source_units"],
                source_claims=source_claims,
            )
            digest = str(packet["claim_coverage"][-1]["excerpt_sha256"])
            packet["claim_coverage"][-1]["excerpt_sha256"] = (
                digest[:18] + digest[19:]
            )
            return packet

        with (
            patch(
                "app.services.analyzer._final_model_input_window",
                new_callable=AsyncMock,
                return_value=self._window(),
            ),
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ),
            patch(
                "app.services.analyzer._structured_artifact",
                new_callable=AsyncMock,
                side_effect=digest_typo_mapper,
            ),
        ):
            model_payload, plan = await _prepare_final_model_payload(
                "run-id",
                payload={
                    "report_data": {
                        "long_narrative": narrative,
                    }
                },
                system="author",
                force_hierarchical=True,
            )

        self.assertTrue(plan["coverage_complete"])
        self.assertEqual(
            plan["covered_claim_count"],
            plan["source_claim_count"],
        )
        self.assertIn(
            "replace_invalid_claim_coverage_digest",
            json.dumps(model_payload, ensure_ascii=False),
        )

    async def test_compact_overflow_builds_bounded_transitive_root(self) -> None:
        tail_marker = "BOUND-ROOT-TAIL-9d4a"
        payload = {
            "report_data": {
                "metrics": [
                    {
                        "name": f"metric-{index}",
                        "value": index,
                        "note": "содержательный факт",
                    }
                    for index in range(180)
                ],
                "tail": "x" * 12_000 + tail_marker,
            }
        }
        target_window = {
            **self._window(),
            "input_token_window": 45_000,
            "input_utf8_window": 45_000,
        }
        worker_window = {
            **self._window(),
            "input_token_window": 150_000,
            "input_utf8_window": 150_000,
        }
        structured_calls: list[dict[str, Any]] = []
        mapper = self._successful_mapper()

        async def capture_mapper(
            *args: Any,
            **kwargs: Any,
        ) -> dict[str, Any]:
            structured_calls.append(copy.deepcopy(kwargs["user_payload"]))
            return await mapper(*args, **kwargs)

        def final_request_bytes(
            candidate: dict[str, Any],
            _envelope: dict[str, Any],
        ) -> int:
            mode = str((candidate.get("long_input_contract") or {}).get("mode") or "")
            if mode == "bounded_transitive_evidence_tree":
                return len(
                    json.dumps(
                        candidate,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                )
            # Force both the canonical and columnar O(N) views through the
            # bounded-root fallback without imposing a source-data cap.
            return 60_000

        with (
            patch(
                "app.services.analyzer._final_model_input_window",
                new_callable=AsyncMock,
                side_effect=[
                    target_window,
                    worker_window,
                    worker_window,
                    worker_window,
                ],
            ),
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ) as save_artifact,
            patch(
                "app.services.analyzer._structured_artifact",
                new_callable=AsyncMock,
                side_effect=capture_mapper,
            ),
        ):
            model_payload, plan = await _prepare_final_model_payload(
                "run-id",
                payload=payload,
                system="author",
                force_hierarchical=True,
                final_request_utf8_bytes=final_request_bytes,
            )

        self.assertEqual(
            plan["mode"],
            "bounded_transitive_evidence_tree",
        )
        root_plan = plan["bounded_root_plan"]
        self.assertTrue(root_plan["coverage_complete"])
        self.assertLessEqual(
            root_plan["model_request_utf8_bytes"],
            target_window["input_utf8_window"],
        )
        root_nodes = model_payload["evidence_digest"]["root_nodes"]
        self.assertEqual(
            sum(int(node["descendant_claim_count"]) for node in root_nodes),
            plan["source_claim_count"],
        )
        self.assertTrue(all("mandatory_fact_table" not in node for node in root_nodes))
        self.assertTrue(
            all("mandatory_fact_provenance_sha256" in node for node in root_nodes)
        )
        self.assertTrue(
            any(
                tail_marker in json.dumps(call, ensure_ascii=False)
                for call in structured_calls
                if "source_nodes" in call
            )
        )
        self.assertTrue(
            any(
                str(call.kwargs.get("artifact_key") or "").startswith(
                    "final_input_bounded_root_tree_"
                )
                for call in save_artifact.await_args_list
            )
        )
        proof_outputs = [
            call.kwargs["output_json"]
            for call in save_artifact.await_args_list
            if "_bounded_root_node_" in str(call.kwargs.get("artifact_key") or "")
        ]
        self.assertTrue(proof_outputs)
        self.assertTrue(
            all(
                isinstance(output.get("fact_provenance"), list)
                and "mandatory_fact_bindings" not in output
                and "mandatory_fact_table" not in output
                and "mandatory_fact_bindings" not in (output.get("model_view") or {})
                and "mandatory_fact_table" not in (output.get("model_view") or {})
                for output in proof_outputs
            )
        )

    def test_bounded_root_rejects_invented_literals_and_missing_excerpts(
        self,
    ) -> None:
        allowed = {
            "node-a": "бренд не найден: 0 из 30 ответов",
            "node-b": "память модели: unknown",
        }
        base = {
            "observations": [
                {
                    "category": "visibility",
                    "statement": "Бренд не найден: 0 из 30 ответов.",
                    "source_node_ids": ["node-a"],
                    "exact_values": ["0 из 30"],
                    "fact_binding_ids": [],
                    "evidence_excerpts": [
                        {
                            "source_node_id": "node-a",
                            "excerpt": "0 из 30",
                        }
                    ],
                    "importance": "critical",
                },
                {
                    "category": "limitation",
                    "statement": "Память модели: unknown.",
                    "source_node_ids": ["node-b"],
                    "exact_values": ["unknown"],
                    "fact_binding_ids": [],
                    "evidence_excerpts": [
                        {
                            "source_node_id": "node-b",
                            "excerpt": "unknown",
                        },
                    ],
                    "importance": "critical",
                },
            ],
            "uncertainties": [
                {
                    "text": "Память модели: unknown.",
                    "source_node_ids": ["node-b"],
                }
            ],
            "report_focus": [],
            "node_coverage": [
                {
                    "source_node_id": node_id,
                    "disposition": "material_observation",
                    "rationale": "Child отражён.",
                }
                for node_id in allowed
            ],
        }
        accepted = _normalize_final_root_summary_packet(
            base,
            allowed_node_text=allowed,
        )
        self.assertEqual(len(accepted["node_coverage"]), 2)

        invented = copy.deepcopy(base)
        invented["observations"][0]["statement"] = "Бренд лидирует: 99,9%."
        invented["observations"][0]["exact_values"] = ["99,9%"]
        with self.assertRaisesRegex(OpenRouterError, "exact value|invented"):
            _normalize_final_root_summary_packet(
                invented,
                allowed_node_text=allowed,
            )

        missing_excerpt = copy.deepcopy(base)
        missing_excerpt["observations"][0]["evidence_excerpts"].pop()
        with self.assertRaisesRegex(OpenRouterError, "omitted a child excerpt"):
            _normalize_final_root_summary_packet(
                missing_excerpt,
                allowed_node_text=allowed,
            )

    def test_bounded_root_rejects_generic_acknowledgement(self) -> None:
        packet = {
            "observations": [
                {
                    "category": "visibility",
                    "statement": "Realweb учтён.",
                    "source_node_ids": ["node-a"],
                    "exact_values": [],
                    "fact_binding_ids": [],
                    "evidence_excerpts": [
                        {
                            "source_node_id": "node-a",
                            "excerpt": "Realweb",
                        }
                    ],
                    "importance": "important",
                }
            ],
            "uncertainties": [],
            "report_focus": [],
            "node_coverage": [
                {
                    "source_node_id": "node-a",
                    "disposition": "supporting_context",
                    "rationale": "Узел отражён.",
                }
            ],
        }

        with self.assertRaisesRegex(
            OpenRouterError,
            "generic acknowledgement",
        ):
            _normalize_final_root_summary_packet(
                packet,
                allowed_node_text={
                    "node-a": "Realweb показывает содержательный результат."
                },
            )

    def test_bounded_root_rejects_cross_child_missing_and_tampered_fact_refs(
        self,
    ) -> None:
        binding_a = self._scalar_fact_binding(
            source_path="/models/chatgpt/mention_rate",
            json_literal='"60%"',
        )
        binding_b = self._scalar_fact_binding(
            source_path="/models/gemini/mention_rate",
            json_literal='"30%"',
        )
        allowed = {
            "node-a": "ChatGPT: mention rate 60%.",
            "node-b": "Gemini: mention rate 30%.",
        }
        ledgers = {"node-a": [binding_a], "node-b": [binding_b]}
        refs = {
            node_id: _final_root_fact_refs(bindings)
            for node_id, bindings in ledgers.items()
        }
        base = {
            "observations": [
                {
                    "category": "visibility",
                    "statement": allowed[node_id],
                    "source_node_ids": [node_id],
                    "exact_values": ["60%" if node_id == "node-a" else "30%"],
                    "fact_binding_ids": refs[node_id],
                    "evidence_excerpts": [
                        {
                            "source_node_id": node_id,
                            "excerpt": allowed[node_id],
                        }
                    ],
                    "importance": "important",
                }
                for node_id in allowed
            ],
            "uncertainties": [],
            "report_focus": [],
            "node_coverage": [
                {
                    "source_node_id": node_id,
                    "disposition": "material_observation",
                    "rationale": "Факт сохранён.",
                }
                for node_id in allowed
            ],
        }
        accepted = _normalize_final_root_summary_packet(
            base,
            allowed_node_text=allowed,
            allowed_node_fact_bindings=ledgers,
        )
        self.assertEqual(
            accepted["observations"][0]["fact_binding_ids"],
            refs["node-a"],
        )

        cross_child = copy.deepcopy(base)
        cross_child["observations"][0]["fact_binding_ids"] = refs["node-b"]
        with self.assertRaisesRegex(OpenRouterError, "mandatory fact binding"):
            _normalize_final_root_summary_packet(
                cross_child,
                allowed_node_text=allowed,
                allowed_node_fact_bindings=ledgers,
            )

        missing = copy.deepcopy(base)
        missing["observations"][0]["fact_binding_ids"] = []
        with self.assertRaisesRegex(OpenRouterError, "mandatory fact binding"):
            _normalize_final_root_summary_packet(
                missing,
                allowed_node_text=allowed,
                allowed_node_fact_bindings=ledgers,
            )

        tampered = copy.deepcopy(base)
        tampered["observations"][0]["fact_binding_ids"] = ["f_tampered"]
        with self.assertRaisesRegex(OpenRouterError, "mandatory fact binding"):
            _normalize_final_root_summary_packet(
                tampered,
                allowed_node_text=allowed,
                allowed_node_fact_bindings=ledgers,
            )

    def test_bounded_root_proof_rejects_tampered_model_view(self) -> None:
        leaves = [
            self._bounded_leaf(
                node_id=f"leaf-{index}",
                index=index,
                semantic_text=f"leaf evidence {index}",
                claim_count=claim_count,
                bindings=[],
            )
            for index, claim_count in enumerate((1, 2))
        ]
        packet = {
            "observations": [
                {
                    "category": "context",
                    "statement": f"leaf evidence {index}",
                    "source_node_ids": [f"leaf-{index}"],
                    "exact_values": [],
                    "fact_binding_ids": [],
                    "evidence_excerpts": [
                        {
                            "source_node_id": f"leaf-{index}",
                            "excerpt": f"leaf evidence {index}",
                        }
                    ],
                    "importance": "supporting",
                }
                for index in range(2)
            ],
            "uncertainties": [],
            "report_focus": [],
            "node_coverage": [
                {
                    "source_node_id": f"leaf-{index}",
                    "disposition": "supporting_context",
                    "rationale": "Leaf сохранён содержательно.",
                }
                for index in range(2)
            ],
        }
        parent = _final_root_parent_node(
            level=1,
            children=leaves,
            packet=packet,
        )
        evil = copy.deepcopy(parent)
        evil["model_view"]["source_node_id"] = "evil-node"
        proof = self._bounded_proof(evil)
        with self.assertRaisesRegex(
            OpenRouterError,
            "model view failed|node id failed",
        ):
            _verify_final_root_tree(
                roots=[evil],
                source_leaves=leaves,
                proof_records={str(evil["node_id"]): proof},
                expected_claim_count=3,
            )

    def test_bounded_root_proof_rejects_tampered_fact_binding(self) -> None:
        binding = self._scalar_fact_binding(
            source_path="/models/chatgpt/mention_rate",
            json_literal='"60%"',
        )
        leaf = self._bounded_leaf(
            node_id="leaf-fact",
            index=0,
            semantic_text="ChatGPT mention rate 60%.",
            claim_count=1,
            bindings=[binding],
        )
        packet = {
            "observations": [
                {
                    "category": "visibility",
                    "statement": "ChatGPT mention rate 60%.",
                    "source_node_ids": ["leaf-fact"],
                    "exact_values": ["60%"],
                    "fact_binding_ids": [],
                    "evidence_excerpts": [
                        {
                            "source_node_id": "leaf-fact",
                            "excerpt": "ChatGPT mention rate 60%.",
                        }
                    ],
                    "importance": "important",
                }
            ],
            "uncertainties": [],
            "report_focus": [],
            "node_coverage": [
                {
                    "source_node_id": "leaf-fact",
                    "disposition": "material_observation",
                    "rationale": "Факт сохранён.",
                }
            ],
        }
        parent = _final_root_parent_node(
            level=1,
            children=[leaf],
            packet=packet,
        )
        proof = self._bounded_proof(parent)
        _verify_final_root_tree(
            roots=[parent],
            source_leaves=[leaf],
            proof_records={str(parent["node_id"]): proof},
            expected_claim_count=1,
        )

        tampered_leaf = copy.deepcopy(leaf)
        tampered_leaf["mandatory_fact_bindings"][0]["json_literal"] = '"90%"'
        with self.assertRaisesRegex(
            OpenRouterError,
            "binding|fact provenance|identity|digest",
        ):
            _verify_final_root_tree(
                roots=[parent],
                source_leaves=[tampered_leaf],
                proof_records={str(parent["node_id"]): proof},
                expected_claim_count=1,
            )

    async def test_reducer_no_fan_in_uses_lossless_terminal_union(self) -> None:
        payload = {
            "report_data": {
                "metrics": [
                    {"name": f"metric-{index}", "value": index} for index in range(300)
                ]
            }
        }
        worker_window = {
            **self._window(),
            "input_token_window": 100_000,
            "input_utf8_window": 100_000,
        }
        final_window = {
            **self._window(),
            "input_token_window": 220_000,
            "input_utf8_window": 220_000,
        }

        async def verbose_mapper(
            *_args: Any,
            **kwargs: Any,
        ) -> dict[str, Any]:
            user_payload = kwargs["user_payload"]
            if "source_units" in user_payload:
                units = user_payload["source_units"]
                packet = self._packet_for_units(
                    units,
                    source_claims=user_payload.get("source_claims"),
                )
                for observation in packet["observations"]:
                    exact_excerpt = str(observation.get("evidence_excerpt") or "")
                    observation["statement"] = (
                        f"Отдельный подтверждённый факт: {exact_excerpt}. " * 8
                    )
                return packet
            if "source_nodes" in user_payload:
                return await self._successful_mapper()(*_args, **kwargs)
            packets = user_payload["evidence_packets"]
            return {
                "observations": [
                    copy.deepcopy(observation)
                    for packet in packets
                    for observation in packet["observations"]
                ],
                "uncertainties": [],
                "report_focus": [],
                "unit_coverage": [
                    copy.deepcopy(item)
                    for packet in packets
                    for item in packet["unit_coverage"]
                ],
                "claim_coverage": [
                    copy.deepcopy(item)
                    for packet in packets
                    for item in packet["claim_coverage"]
                ],
            }

        with (
            patch(
                "app.services.analyzer._final_model_input_window",
                new_callable=AsyncMock,
                side_effect=[
                    final_window,
                    worker_window,
                    worker_window,
                    worker_window,
                ],
            ),
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ) as save_artifact,
            patch(
                "app.services.analyzer._structured_artifact",
                new_callable=AsyncMock,
                side_effect=verbose_mapper,
            ) as structured_artifact,
        ):
            model_payload, plan = await _prepare_final_model_payload(
                "run-id",
                payload=payload,
                system="author",
                force_hierarchical=True,
            )

        self.assertEqual(
            plan["terminal_reducer_mode"],
            "deterministic_lossless_union",
        )
        self.assertFalse(
            any(
                "evidence_packets" in (call.kwargs.get("user_payload") or {})
                for call in structured_artifact.await_args_list
            )
        )
        self.assertIn(
            plan["mode"],
            {
                "hierarchical_evidence_tree_compact_ledger",
                "bounded_transitive_evidence_tree",
            },
        )
        terminal_ledgers = [
            call.kwargs["output_json"]
            for call in save_artifact.await_args_list
            if "_terminal_ledger_" in str(call.kwargs.get("artifact_key") or "")
        ]
        self.assertEqual(len(terminal_ledgers), 1)
        terminal_evidence = terminal_ledgers[0]["evidence_digest"]
        self.assertEqual(
            len(terminal_evidence["observations"]),
            plan["source_unit_count"],
        )
        self.assertEqual(
            len(terminal_evidence["unit_coverage"]),
            plan["source_unit_count"],
        )
        self.assertEqual(
            sorted(
                str(item["source_unit_id"])
                for item in terminal_evidence["unit_coverage"]
            ),
            sorted(
                str(source_unit_id)
                for item in terminal_evidence["observations"]
                for source_unit_id in item["source_unit_ids"]
            ),
        )
        self.assertEqual(
            model_payload["long_input_contract"]["source_unit_count"],
            plan["source_unit_count"],
        )
        self.assertLessEqual(
            plan["model_request_utf8_bytes"],
            plan["input_utf8_window"],
        )

    async def test_empty_mapper_and_unobserved_units_fail_closed(self) -> None:
        empty_packet = {
            "observations": [],
            "uncertainties": [],
            "report_focus": [],
            "unit_coverage": [],
        }

        async def coverage_only(
            *args: Any,
            **kwargs: Any,
        ) -> dict[str, Any]:
            units = kwargs["user_payload"]["source_units"]
            return {
                **empty_packet,
                "unit_coverage": [
                    {
                        "source_unit_id": str(unit["source_unit_id"]),
                        "disposition": "supporting_context",
                        "rationale": "Заявлено без observation.",
                    }
                    for unit in units
                ],
            }

        for response, expected in (
            (empty_packet, "does not account for every source unit"),
            (coverage_only, "without an explicit observation"),
        ):
            with self.subTest(expected=expected):
                side_effect = response if callable(response) else None
                return_value = None if side_effect else response
                with (
                    patch(
                        "app.services.analyzer._final_model_input_window",
                        new_callable=AsyncMock,
                        return_value=self._window(),
                    ),
                    patch(
                        "app.services.analyzer._save_artifact",
                        new_callable=AsyncMock,
                    ),
                    patch(
                        "app.services.analyzer._structured_artifact",
                        new_callable=AsyncMock,
                        side_effect=side_effect,
                        return_value=return_value,
                    ),
                ):
                    with self.assertRaisesRegex(OpenRouterError, expected):
                        await _prepare_final_model_payload(
                            "run-id",
                            payload=self._oversized_payload(),
                            system="author",
                        )

    async def test_mapper_cannot_hide_metric_scalar(self) -> None:
        with (
            patch(
                "app.services.analyzer._final_model_input_window",
                new_callable=AsyncMock,
                return_value=self._window(),
            ),
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ),
            patch(
                "app.services.analyzer._structured_artifact",
                new_callable=AsyncMock,
                side_effect=self._successful_mapper(hide_values=True),
            ),
        ):
            model_payload, _ = await _prepare_final_model_payload(
                "run-id",
                payload=self._oversized_payload(),
                system="author",
            )

        self.assertFalse(
            any(
                observation["exact_values"]
                for observation in model_payload["evidence_digest"]["observations"]
            )
        )
        passthrough = {
            item["source_path"]: item
            for item in model_payload["deterministic_passthrough"]["values"]
        }
        metric = passthrough["/report_data/critical_metric"]
        self.assertEqual(metric["value"], 73)
        self.assertEqual(metric["value_type"], "number")
        self.assertEqual(metric["json_literal"], "73")
        self.assertEqual(
            passthrough["/report_data/data_state"]["value"],
            "unknown",
        )
        self.assertIs(
            passthrough["/report_data/eligible"]["value"],
            True,
        )
        self.assertIsNone(passthrough["/report_data/optional_value"]["value"])

    async def test_mapper_and_reducer_use_their_own_exact_envelopes(self) -> None:
        envelopes = {
            PROCESSING_MODEL: {
                "version": "test",
                "resolution": "test-processing",
                "context_length": 36_000,
                "max_completion_tokens": 20_000,
            },
            ANALYSIS_MODEL: {
                "version": "test",
                "resolution": "test-analysis",
                "context_length": 80_000,
                "max_completion_tokens": 16_000,
            },
        }

        async def envelope_for(model: str) -> dict[str, Any]:
            return copy.deepcopy(envelopes[model])

        with (
            patch(
                "app.services.analyzer.model_output_envelope",
                new_callable=AsyncMock,
                side_effect=envelope_for,
            ) as envelope_mock,
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ) as save_artifact,
            patch(
                "app.services.analyzer._structured_artifact",
                new_callable=AsyncMock,
                side_effect=self._successful_mapper(),
            ),
        ):
            _model_payload, plan = await _prepare_final_model_payload(
                "run-id",
                payload=self._oversized_payload(),
                system="author",
            )

        requested_models = [
            call.args[0] if call.args else call.kwargs["model"]
            for call in envelope_mock.await_args_list
        ]
        self.assertIn(PROCESSING_MODEL, requested_models)
        self.assertIn(ANALYSIS_MODEL, requested_models)
        self.assertLess(
            plan["mapper_request_window_utf8_bytes"],
            plan["reducer_request_window_utf8_bytes"],
        )
        self.assertLessEqual(
            plan["mapper_max_request_utf8_bytes"],
            plan["mapper_request_window_utf8_bytes"],
        )
        self.assertLessEqual(
            plan["reducer_max_request_utf8_bytes"],
            plan["reducer_request_window_utf8_bytes"],
        )
        manifests = [
            call.kwargs["output_json"]
            for call in save_artifact.await_args_list
            if "_manifest_" in call.kwargs.get("artifact_key", "")
        ]
        self.assertEqual(len(manifests), 1)
        self.assertLess(manifests[0]["map_target_chars"], 32_000)

    async def test_oversized_semantic_evidence_is_bounded_for_provider(
        self,
    ) -> None:
        full_only_marker = "FULL-EVIDENCE-ONLY-TAIL-9fd0f4"
        review_input = {
            "evidence_document": {
                "report_data": {
                    "long_narrative": "x" * 260_000 + full_only_marker,
                    "critical_metric": 73,
                },
                "selected_answer_context": [],
                "answer_selection_manifest": {"digest": "selection"},
            },
            "metric_availability_contract": [],
            "candidate_report": {
                "headline": "Итог",
                "headline_emphasis": [],
                "sections": [{"heading": "Раздел", "body": "Текст"}],
                "actions": [{"title": "Шаг", "why": "Причина"}],
                "limitations": [],
            },
            "deterministic_precheck_errors": [],
        }
        envelopes = {
            PROCESSING_MODEL: {
                "version": "test",
                "resolution": "test-processing",
                "context_length": 36_000,
                "max_completion_tokens": 20_000,
            },
            ANALYSIS_MODEL: {
                "version": "test",
                "resolution": "test-analysis",
                "context_length": 80_000,
                "max_completion_tokens": 16_000,
            },
            REPORT_SEMANTIC_MODEL: {
                "version": "test",
                "resolution": "test-semantic",
                "context_length": 80_000,
                "max_completion_tokens": 16_000,
            },
        }

        async def envelope_for(model: str) -> dict[str, Any]:
            return copy.deepcopy(envelopes[model])

        with (
            patch(
                "app.services.analyzer.model_output_envelope",
                new_callable=AsyncMock,
                side_effect=envelope_for,
            ),
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ),
            patch(
                "app.services.analyzer._structured_artifact",
                new_callable=AsyncMock,
                side_effect=self._successful_mapper(hide_values=True),
            ),
        ):
            model_context, plan = await _bounded_semantic_model_evidence_context(
                "run-id",
                review_input=review_input,
                attempt=1,
            )

        provider_probe = copy.deepcopy(review_input)
        provider_probe["model_evidence_context"] = model_context
        provider_bytes = semantic_provider_request_utf8_bytes(
            provider_probe,
            attempt=1,
            model_envelope=plan["model_envelope"],
        )
        provider_payload = semantic_provider_payload(provider_probe)
        provider_serialized = json.dumps(
            provider_payload,
            ensure_ascii=False,
        )

        self.assertEqual(
            plan["mode"],
            "hierarchical_evidence_tree_compact_ledger",
        )
        self.assertLessEqual(provider_bytes, plan["input_utf8_window"])
        self.assertEqual(
            provider_bytes,
            plan["model_request_utf8_bytes"],
        )
        self.assertNotEqual(
            provider_payload["evidence_document"],
            review_input["evidence_document"],
        )
        self.assertNotIn(full_only_marker, provider_serialized)
        self.assertTrue(
            review_input["evidence_document"]["report_data"]["long_narrative"].endswith(
                full_only_marker
            )
        )

    async def test_semantic_wrapper_overhead_forces_near_window_partition(
        self,
    ) -> None:
        review_input = {
            "evidence_document": {
                "report_data": {"long_narrative": "x" * 45_000},
                "selected_answer_context": [],
                "answer_selection_manifest": {},
            },
            "metric_availability_contract": [],
            "candidate_report": {
                "headline": "Итог",
                "headline_emphasis": [],
                "sections": [],
                "actions": [],
                "limitations": [],
            },
            "deterministic_precheck_errors": [],
        }
        envelopes = {
            PROCESSING_MODEL: {
                "version": "test",
                "resolution": "test-processing",
                "context_length": 36_000,
                "max_completion_tokens": 20_000,
            },
            ANALYSIS_MODEL: {
                "version": "test",
                "resolution": "test-analysis",
                "context_length": 80_000,
                "max_completion_tokens": 16_000,
            },
            REPORT_SEMANTIC_MODEL: {
                "version": "test",
                "resolution": "test-semantic",
                "context_length": 68_000,
                "max_completion_tokens": 16_000,
            },
        }

        async def envelope_for(model: str) -> dict[str, Any]:
            return copy.deepcopy(envelopes[model])

        direct_probe = copy.deepcopy(review_input)
        direct_probe["model_evidence_context"] = copy.deepcopy(
            review_input["evidence_document"]
        )
        expected_window = 68_000 - 16_000
        evidence_bytes = len(
            json.dumps(
                review_input["evidence_document"],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        direct_request_bytes = semantic_provider_request_utf8_bytes(
            direct_probe,
            attempt=1,
            model_envelope=envelopes[REPORT_SEMANTIC_MODEL],
        )
        self.assertLess(evidence_bytes, expected_window)
        self.assertGreater(direct_request_bytes, expected_window)

        with (
            patch(
                "app.services.analyzer.model_output_envelope",
                new_callable=AsyncMock,
                side_effect=envelope_for,
            ),
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ),
            patch(
                "app.services.analyzer._structured_artifact",
                new_callable=AsyncMock,
                side_effect=self._successful_mapper(hide_values=True),
            ),
        ):
            _model_context, plan = await _bounded_semantic_model_evidence_context(
                "run-id",
                review_input=review_input,
                attempt=1,
            )

        self.assertEqual(plan["input_utf8_window"], expected_window)
        self.assertEqual(plan["mode"], "hierarchical_evidence_tree")
        self.assertLessEqual(
            plan["model_request_utf8_bytes"],
            plan["input_utf8_window"],
        )


class FinalReportPreflightTests(unittest.IsolatedAsyncioTestCase):
    async def test_legacy_local_budget_cannot_reject_a_final_document(
        self,
    ) -> None:
        payload = {
            "report_data": {"brand": {"name": "Example"}},
            "answer_selection_manifest": {"digest": "selection"},
            "answer_corpus_manifest": {"digest": "corpus-digest"},
            "selected_full_answers": [{"answer_text": "я" * 100}],
        }
        candidate = {
            "headline": "Итог",
            "headline_emphasis": [],
            "sections": [{"heading": "Раздел", "body": "Текст"}],
            "actions": [{"title": "Действие", "why": "Причина"}],
        }
        with (
            patch.object(settings, "FINAL_INPUT_TOKEN_BUDGET", 1),
            patch(
                "app.services.analyzer._final_report_payload",
                return_value=payload,
            ),
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ) as save_artifact,
            patch(
                "app.services.analyzer._artifact_output",
                new_callable=AsyncMock,
            ) as artifact_output,
            patch(
                "app.services.analyzer.chat_continuable_structured",
                new_callable=AsyncMock,
                return_value=SimpleNamespace(
                    parsed=candidate,
                    text=json.dumps(candidate, ensure_ascii=False),
                    usage={},
                ),
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
            patch(
                "app.services.analyzer._edit_final_report_language",
                new=AsyncMock(side_effect=lambda _run_id, **kwargs: kwargs["report"]),
            ),
        ):
            result = await _final_report(
                "run-id",
                payload["report_data"],
                {"manifest": {"digest": "corpus-digest"}, "answers": [{}]},
            )

        self.assertEqual(result, candidate)
        self.assertGreaterEqual(artifact_output.await_count, 1)
        self.assertEqual(final_chat.await_count, 1)
        preflight_writes = [
            call.kwargs
            for call in save_artifact.await_args_list
            if call.kwargs.get("artifact_key") == "final_report_preflight"
        ]
        self.assertEqual(preflight_writes[-1]["status"], "completed")
        self.assertTrue(preflight_writes[-1]["output_json"]["accepted"])
        self.assertFalse(
            preflight_writes[-1]["output_json"]["legacy_budget_would_accept"]
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

    def test_reviewer_outage_is_fail_soft_but_audit_corruption_is_not(self) -> None:
        self.assertTrue(
            _semantic_reviewer_failure_is_fail_soft(
                OpenRouterError("semantic reviewer provider unavailable")
            )
        )
        self.assertFalse(
            _semantic_reviewer_failure_is_fail_soft(
                OpenRouterAuditCheckpointError(
                    "Stored semantic physical receipt is corrupt",
                    event={},
                )
            )
        )

    async def test_structure_repair_is_repreflighted_before_provider_call(
        self,
    ) -> None:
        payload = self._payload()
        rejected = {
            "headline": "Неполный ответ",
            "sections": [],
            "actions": [],
        }
        repaired = self._candidate("Исправлено")
        prepared_repair = {
            "long_input_contract": {"mode": "hierarchical_evidence_tree"},
            "evidence_digest": {"marker": "prepared-structure-repair"},
        }
        with (
            patch(
                "app.services.analyzer._artifact_output",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ),
            patch(
                "app.services.analyzer._prepare_final_model_payload",
                new_callable=AsyncMock,
                return_value=(prepared_repair, {"mode": "hierarchical"}),
            ) as prepare,
            patch(
                "app.services.analyzer._final_report_structured_attempt",
                new_callable=AsyncMock,
                side_effect=[
                    SimpleNamespace(parsed=rejected, text="bad", usage={}),
                    SimpleNamespace(parsed=repaired, text="good", usage={}),
                ],
            ) as attempt,
        ):
            candidate, _raw, _usage = await _final_report_author_candidate(
                "run-id",
                system="author",
                payload=payload,
            )

        self.assertEqual(candidate, repaired)
        prepare.assert_awaited_once()
        repair_source = prepare.await_args.kwargs["payload"]
        self.assertEqual(repair_source["rejected_report"], rejected)
        self.assertIn("structure_validation_errors_to_fix", repair_source)
        self.assertIs(
            attempt.await_args_list[1].kwargs["user_payload"],
            prepared_repair,
        )

    async def test_semantic_repair_is_repreflighted_before_provider_call(
        self,
    ) -> None:
        payload = self._payload()
        candidate = self._candidate("Первый кандидат")
        repaired = self._candidate("Семантика исправлена")
        initial_prepared = {"prepared": "initial"}
        repair_prepared = {
            "long_input_contract": {"mode": "hierarchical_evidence_tree"},
            "evidence_digest": {"marker": "prepared-semantic-repair"},
        }
        with (
            patch(
                "app.services.analyzer._final_report_payload",
                return_value=payload,
            ),
            patch(
                "app.services.analyzer._prepare_final_model_payload",
                new_callable=AsyncMock,
                side_effect=[
                    (initial_prepared, self._window_plan()),
                    (repair_prepared, self._window_plan()),
                ],
            ) as prepare,
            patch(
                "app.services.analyzer._artifact_output",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ),
            patch(
                "app.services.analyzer._final_report_author_candidate",
                new_callable=AsyncMock,
                return_value=(candidate, "raw", {}),
            ),
            patch(
                "app.services.analyzer._final_report_semantic_review_artifact",
                new_callable=AsyncMock,
                side_effect=[
                    {"verdict": "revise"},
                    {"verdict": "pass"},
                ],
            ),
            patch(
                "app.services.analyzer.report_semantic_blockers",
                side_effect=[["Исправьте вывод."], []],
            ),
            patch(
                "app.services.analyzer._final_report_structured_attempt",
                new_callable=AsyncMock,
                return_value=SimpleNamespace(
                    parsed=repaired,
                    text="repair",
                    usage={},
                ),
            ) as attempt,
        ):
            result = await _final_report(
                "run-id",
                payload["report_data"],
                {"manifest": {"digest": "corpus"}, "answers": [{}]},
            )

        self.assertEqual(result, repaired)
        self.assertEqual(prepare.await_count, 2)
        repair_source = prepare.await_args_list[1].kwargs["payload"]
        self.assertEqual(repair_source["rejected_report"], candidate)
        self.assertEqual(
            repair_source["semantic_review_to_fix"]["verdict"],
            "revise",
        )
        self.assertIs(attempt.await_args.kwargs["user_payload"], repair_prepared)

    @staticmethod
    def _window_plan() -> dict[str, Any]:
        return {
            "input_token_window": 192_000,
            "input_utf8_window": 192_000,
            "mode": "direct",
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
                "app.services.analyzer.chat_continuable_structured",
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
            patch(
                "app.services.analyzer._edit_final_report_language",
                new=AsyncMock(side_effect=lambda _run_id, **kwargs: kwargs["report"]),
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
            if call.kwargs.get("artifact_key") == FINAL_REPORT_AUTHOR_ARTIFACT_KEY
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
                "app.services.analyzer.chat_continuable_structured",
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
            if call.kwargs.get("artifact_key") == FINAL_REPORT_AUTHOR_ARTIFACT_KEY
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
                "app.services.analyzer.chat_continuable_structured",
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
            patch(
                "app.services.analyzer._edit_final_report_language",
                new=AsyncMock(side_effect=lambda _run_id, **kwargs: kwargs["report"]),
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
                call.kwargs.get("artifact_key") == FINAL_REPORT_AUTHOR_ARTIFACT_KEY
                for call in save_artifact.await_args_list
            )
        )

    async def test_semantic_block_requests_bounded_repair_without_cache_invalidation(
        self,
    ) -> None:
        payload = self._payload()
        blocked_candidate = self._candidate("Заблокированный кандидат")
        replacement_candidate = self._candidate("Новый кандидат")
        with (
            patch(
                "app.services.analyzer._final_report_payload",
                return_value=payload,
            ),
            patch(
                "app.services.analyzer._prepare_final_model_payload",
                new_callable=AsyncMock,
                side_effect=[
                    ({"prepared": "initial"}, self._window_plan()),
                    ({"prepared": "repair"}, self._window_plan()),
                ],
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
                "app.services.analyzer._final_report_author_candidate",
                new_callable=AsyncMock,
                return_value=(blocked_candidate, "blocked", {"candidate": 1}),
            ) as author,
            patch(
                "app.services.analyzer._final_report_structured_attempt",
                new_callable=AsyncMock,
                return_value=SimpleNamespace(
                    parsed=replacement_candidate,
                    text="replacement",
                    usage={"candidate": 2},
                ),
            ) as repair,
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
            patch(
                "app.services.analyzer._edit_final_report_language",
                new=AsyncMock(side_effect=lambda _run_id, **kwargs: kwargs["report"]),
            ),
        ):
            result = await _final_report(
                "run-id",
                payload["report_data"],
                {"manifest": {"digest": "corpus"}, "answers": [{}]},
            )

        self.assertEqual(result, replacement_candidate)
        author.assert_awaited_once()
        repair.assert_awaited_once()
        self.assertEqual(semantic_review.await_count, 2)
        reviewed_candidates = [
            call.kwargs["candidate_report"] for call in semantic_review.await_args_list
        ]
        self.assertEqual(
            reviewed_candidates,
            [blocked_candidate, replacement_candidate],
        )
        author_failures = [
            call.kwargs
            for call in save_artifact.await_args_list
            if call.kwargs.get("artifact_key") == FINAL_REPORT_AUTHOR_ARTIFACT_KEY
            and call.kwargs.get("status") == "failed"
        ]
        self.assertEqual(author_failures, [])
        admissions = [
            call.kwargs["output_json"]
            for call in save_artifact.await_args_list
            if str(call.kwargs.get("artifact_key") or "").startswith(
                "final_report_semantic_admission_"
            )
        ]
        self.assertEqual(admissions[-1]["decision"], "pass")
        self.assertFalse(admissions[-1]["reviewer_has_hard_veto"])

    async def test_semantic_reviewer_outage_publishes_deterministic_candidate_degraded(
        self,
    ) -> None:
        payload = self._payload()
        candidate = self._candidate("Детерминированно безопасный кандидат")
        with (
            patch(
                "app.services.analyzer._final_report_payload",
                return_value=payload,
            ),
            patch(
                "app.services.analyzer._prepare_final_model_payload",
                new_callable=AsyncMock,
                return_value=({"prepared": "initial"}, self._window_plan()),
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
                "app.services.analyzer._final_report_author_candidate",
                new_callable=AsyncMock,
                return_value=(candidate, "raw", {}),
            ),
            patch(
                "app.services.analyzer._final_report_structured_attempt",
                new_callable=AsyncMock,
            ) as repair,
            patch(
                "app.services.analyzer._final_report_semantic_review_artifact",
                new_callable=AsyncMock,
                side_effect=_FinalSemanticReviewerUnavailable("provider outage"),
            ),
            patch(
                "app.services.analyzer._edit_final_report_language",
                new=AsyncMock(side_effect=lambda _run_id, **kwargs: kwargs["report"]),
            ),
        ):
            result = await _final_report(
                "run-id",
                payload["report_data"],
                {"manifest": {"digest": "corpus"}, "answers": [{}]},
            )

        self.assertEqual(result, candidate)
        repair.assert_not_awaited()
        admissions = [
            call.kwargs["output_json"]
            for call in save_artifact.await_args_list
            if str(call.kwargs.get("artifact_key") or "").startswith(
                "final_report_semantic_admission_"
            )
        ]
        self.assertEqual(admissions[-1]["decision"], "degraded_safe")
        self.assertEqual(admissions[-1]["reviewer_state"], "unavailable")
        self.assertIn(
            "semantic_reviewer_unavailable",
            admissions[-1]["degraded_reason_codes"],
        )

    async def test_semantic_evidence_preparation_outage_is_degraded_not_terminal(
        self,
    ) -> None:
        payload = self._payload()
        candidate = self._candidate("Безопасный кандидат до semantic preflight")
        with (
            patch(
                "app.services.analyzer._final_report_payload",
                return_value=payload,
            ),
            patch(
                "app.services.analyzer._prepare_final_model_payload",
                new_callable=AsyncMock,
                return_value=({"prepared": "initial"}, self._window_plan()),
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
                "app.services.analyzer._final_report_author_candidate",
                new_callable=AsyncMock,
                return_value=(candidate, "raw", {}),
            ),
            patch(
                "app.services.analyzer._final_report_structured_attempt",
                new_callable=AsyncMock,
            ) as repair,
            patch(
                "app.services.analyzer._bounded_semantic_model_evidence_context",
                new_callable=AsyncMock,
                side_effect=OpenRouterError("semantic mapper provider outage"),
            ) as semantic_preflight,
            patch(
                "app.services.analyzer.review_final_report_semantics",
                new_callable=AsyncMock,
            ) as semantic_provider,
            patch(
                "app.services.analyzer.report_semantic_blockers",
            ) as semantic_blockers,
            patch(
                "app.services.analyzer._edit_final_report_language",
                new=AsyncMock(side_effect=lambda _run_id, **kwargs: kwargs["report"]),
            ),
        ):
            result = await _final_report(
                "run-id",
                payload["report_data"],
                {"manifest": {"digest": "corpus"}, "answers": [{}]},
            )

        self.assertEqual(result, candidate)
        semantic_preflight.assert_awaited_once()
        semantic_provider.assert_not_awaited()
        semantic_blockers.assert_not_called()
        repair.assert_not_awaited()
        admissions = [
            call.kwargs["output_json"]
            for call in save_artifact.await_args_list
            if str(call.kwargs.get("artifact_key") or "").startswith(
                "final_report_semantic_admission_"
            )
        ]
        self.assertEqual(admissions[-1]["decision"], "degraded_safe")
        self.assertEqual(admissions[-1]["reviewer_state"], "unavailable")
        self.assertIn(
            "semantic_reviewer_unavailable",
            admissions[-1]["degraded_reason_codes"],
        )

    async def test_reviewer_outage_after_finding_uses_latest_grounded_repair_degraded(
        self,
    ) -> None:
        payload = self._payload()
        candidate = self._candidate("Кандидат с неподтверждённым фактом")
        repair_one = self._candidate("Непроверенная правка один")
        repair_two = self._candidate("Непроверенная правка два")
        with (
            patch(
                "app.services.analyzer._final_report_payload",
                return_value=payload,
            ),
            patch(
                "app.services.analyzer._prepare_final_model_payload",
                new_callable=AsyncMock,
                side_effect=[
                    ({"prepared": "initial"}, self._window_plan()),
                    ({"prepared": "repair-one"}, self._window_plan()),
                    ({"prepared": "repair-two"}, self._window_plan()),
                ],
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
                "app.services.analyzer._final_report_author_candidate",
                new_callable=AsyncMock,
                return_value=(candidate, "initial", {}),
            ),
            patch(
                "app.services.analyzer._final_report_structured_attempt",
                new_callable=AsyncMock,
                side_effect=[
                    SimpleNamespace(parsed=repair_one, text="one", usage={}),
                    SimpleNamespace(parsed=repair_two, text="two", usage={}),
                ],
            ),
            patch(
                "app.services.analyzer._final_report_semantic_review_artifact",
                new_callable=AsyncMock,
                side_effect=[
                    {"verdict": "block"},
                    _FinalSemanticReviewerUnavailable("provider outage"),
                    _FinalSemanticReviewerUnavailable("provider outage"),
                ],
            ) as semantic_review,
            patch(
                "app.services.analyzer.report_semantic_blockers",
                return_value=["Неподтверждённое числовое утверждение."],
            ) as semantic_blockers,
            patch(
                "app.services.analyzer._edit_final_report_language",
                new=AsyncMock(side_effect=lambda _run_id, **kwargs: kwargs["report"]),
            ) as editor,
        ):
            result = await _final_report(
                "run-id",
                payload["report_data"],
                {"manifest": {"digest": "corpus"}, "answers": [{}]},
            )

        self.assertEqual(result, repair_two)
        self.assertEqual(semantic_review.await_count, 3)
        semantic_blockers.assert_called_once()
        editor.assert_awaited_once()
        admissions = [
            call.kwargs["output_json"]
            for call in save_artifact.await_args_list
            if str(call.kwargs.get("artifact_key") or "").startswith(
                "final_report_semantic_admission_"
            )
        ]
        self.assertEqual(admissions[-1]["decision"], "degraded_safe")
        self.assertIsNotNone(admissions[-1]["selected_report_sha256"])
        self.assertTrue(
            admissions[-1]["fallback_to_deterministically_safe_candidate"]
        )
        self.assertIn(
            "semantic_reviewer_unavailable_after_findings",
            admissions[-1]["degraded_reason_codes"],
        )
        self.assertIn(
            "semantic_safe_rollback",
            admissions[-1]["degraded_reason_codes"],
        )

    async def test_unresolved_reviewer_findings_publish_latest_grounded_candidate_degraded(
        self,
    ) -> None:
        payload = self._payload()
        payload["report_data"]["discovery"] = {"mention_rate": 66.7}
        candidate = self._candidate("Первый безопасный кандидат: 66,7%")
        repair_one = self._candidate("Первая безопасная правка: 66,7%")
        repair_two = self._candidate("Вторая безопасная правка: 66,7%")
        with (
            patch(
                "app.services.analyzer._final_report_payload",
                return_value=payload,
            ),
            patch(
                "app.services.analyzer._prepare_final_model_payload",
                new_callable=AsyncMock,
                side_effect=[
                    ({"prepared": "initial"}, self._window_plan()),
                    ({"prepared": "repair-one"}, self._window_plan()),
                    ({"prepared": "repair-two"}, self._window_plan()),
                ],
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
                "app.services.analyzer._final_report_author_candidate",
                new_callable=AsyncMock,
                return_value=(candidate, "initial", {}),
            ),
            patch(
                "app.services.analyzer._final_report_structured_attempt",
                new_callable=AsyncMock,
                side_effect=[
                    SimpleNamespace(parsed=repair_one, text="one", usage={}),
                    SimpleNamespace(parsed=repair_two, text="two", usage={}),
                ],
            ) as repair,
            patch(
                "app.services.analyzer._final_report_semantic_review_artifact",
                new_callable=AsyncMock,
                side_effect=[
                    {"verdict": "block"},
                    {"verdict": "block"},
                    {"verdict": "block"},
                ],
            ) as semantic_review,
            patch(
                "app.services.analyzer.report_semantic_blockers",
                side_effect=[
                    ["Модель не смогла доказать утверждение."],
                    ["Модель не смогла доказать утверждение."],
                    ["Модель не смогла доказать утверждение."],
                ],
            ),
            patch(
                "app.services.analyzer._edit_final_report_language",
                new=AsyncMock(side_effect=lambda _run_id, **kwargs: kwargs["report"]),
            ),
        ):
            result = await _final_report(
                "run-id",
                payload["report_data"],
                {"manifest": {"digest": "corpus"}, "answers": [{}]},
            )

        self.assertEqual(result, repair_two)
        self.assertEqual(repair.await_count, 2)
        self.assertEqual(semantic_review.await_count, 3)
        admissions = [
            call.kwargs["output_json"]
            for call in save_artifact.await_args_list
            if str(call.kwargs.get("artifact_key") or "").startswith(
                "final_report_semantic_admission_"
            )
        ]
        self.assertEqual(admissions[-1]["decision"], "degraded_safe")
        self.assertEqual(admissions[-1]["repair_attempts_used"], 2)
        self.assertTrue(
            admissions[-1]["fallback_to_deterministically_safe_candidate"]
        )
        self.assertIsNotNone(admissions[-1]["selected_report_sha256"])
        self.assertIn(
            "Модель не смогла доказать утверждение.",
            admissions[-1]["reviewer_findings"],
        )
        self.assertIn(
            "semantic_repair_exhausted",
            admissions[-1]["degraded_reason_codes"],
        )
        self.assertIn(
            "semantic_safe_rollback",
            admissions[-1]["degraded_reason_codes"],
        )
        final_writes = [
            call.kwargs
            for call in save_artifact.await_args_list
            if call.kwargs.get("artifact_key") == "final_report"
        ]
        self.assertEqual(final_writes[-1]["status"], "completed")

    async def test_unsupported_exact_percentage_blocks_after_bounded_repairs(
        self,
    ) -> None:
        payload = self._payload()
        candidate = self._candidate("Неподтверждённое значение: 777%")
        repair_one = self._candidate("Первая правка всё ещё утверждает 777%")
        repair_two = self._candidate("Вторая правка всё ещё утверждает 777%")
        with (
            patch(
                "app.services.analyzer._final_report_payload",
                return_value=payload,
            ),
            patch(
                "app.services.analyzer._prepare_final_model_payload",
                new_callable=AsyncMock,
                side_effect=[
                    ({"prepared": "initial"}, self._window_plan()),
                    ({"prepared": "repair-one"}, self._window_plan()),
                    ({"prepared": "repair-two"}, self._window_plan()),
                ],
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
                "app.services.analyzer._final_report_author_candidate",
                new_callable=AsyncMock,
                return_value=(candidate, "initial", {}),
            ),
            patch(
                "app.services.analyzer._final_report_structured_attempt",
                new_callable=AsyncMock,
                side_effect=[
                    SimpleNamespace(parsed=repair_one, text="one", usage={}),
                    SimpleNamespace(parsed=repair_two, text="two", usage={}),
                ],
            ) as repair,
            patch(
                "app.services.analyzer._final_report_semantic_review_artifact",
                new_callable=AsyncMock,
                return_value={"verdict": "pass"},
            ) as semantic_review,
            patch(
                "app.services.analyzer.report_semantic_blockers",
                return_value=[],
            ),
            patch(
                "app.services.analyzer._edit_final_report_language",
                new_callable=AsyncMock,
            ) as editor,
        ):
            with self.assertRaisesRegex(
                OpenRouterError,
                "Final report semantic admission failed",
            ):
                await _final_report(
                    "run-id",
                    payload["report_data"],
                    {"manifest": {"digest": "corpus"}, "answers": [{}]},
                )

        self.assertEqual(repair.await_count, 2)
        semantic_review.assert_awaited_once()
        editor.assert_not_awaited()
        admissions = [
            call.kwargs["output_json"]
            for call in save_artifact.await_args_list
            if str(call.kwargs.get("artifact_key") or "").startswith(
                "final_report_semantic_admission_"
            )
        ]
        self.assertEqual(admissions[-1]["decision"], "block")
        self.assertIsNone(admissions[-1]["selected_report_sha256"])
        self.assertTrue(
            any(
                "неподтверждённый точный" in error
                for error in admissions[-1]["deterministic_errors"]
            )
        )
        final_writes = [
            call.kwargs
            for call in save_artifact.await_args_list
            if call.kwargs.get("artifact_key") == "final_report"
        ]
        self.assertEqual(final_writes[-1]["status"], "failed")


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
                        model = panel.model if mode == "web" else panel.memory_model
                        if model is None:
                            continue
                        is_last = (
                            prompt_index == 1
                            and mode == "memory"
                            and panel.key == "claude"
                        )
                        answer_text = f"Ответ {prompt_index}/{panel.key}/{mode} " + (
                            "LAST-DB-SENTINEL" if is_last else "END"
                        )
                        raw_sha256 = hashlib.sha256(
                            answer_text.encode("utf-8")
                        ).hexdigest()
                        usage_json, citations_json = _attested_panel_usage(
                            prompt_text=prompt.text,
                            mode=mode,
                            provider_key=panel.key,
                            model=model,
                            response_text=answer_text,
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
                "panel_metric_coverage_admission": (
                    coverage_admission := build_panel_metric_coverage_admission(
                        expected_cells=expected_cells,
                        observed_rows=critic_rows,
                    )
                ),
                "panel_metric_coverage_admission_sha256": coverage_admission[
                    "admission_sha256"
                ],
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

    async def test_terminal_failed_provider_lane_is_metadata_only_not_missing(
        self,
    ) -> None:
        run_id = f"test-final-partial-provider-{uuid.uuid4()}"
        async with SessionLocal() as session:
            session.add(
                Run(
                    id=run_id,
                    domain="example.com",
                    status=RunStatus.analyzing,
                    config_json={},
                )
            )
            for prompt_index in range(9):
                prompt = VisibilityPrompt(
                    run_id=run_id,
                    prompt_key=f"prompt-{prompt_index}",
                    intent_class="I",
                    role="unbranded_discovery",
                    text=f"Сценарий {prompt_index}",
                    rationale="Проверка допустимой недоступности провайдера.",
                    sequence=prompt_index + 1,
                )
                session.add(prompt)
                await session.flush()
                for mode in ("web", "memory"):
                    for panel in panel_models():
                        model = panel.model if mode == "web" else panel.memory_model
                        if model is None:
                            continue
                        failed_lane = mode == "memory" and panel.key == "claude"
                        answer_text = (
                            ""
                            if failed_lane
                            else f"Ответ {prompt_index}/{panel.key}/{mode} END"
                        )
                        if failed_lane:
                            usage_json, citations_json = {}, []
                        else:
                            usage_json, citations_json = _attested_panel_usage(
                                prompt_text=prompt.text,
                                mode=mode,
                                provider_key=panel.key,
                                model=model,
                                response_text=answer_text,
                            )
                        answer = ModelAnswer(
                            run_id=run_id,
                            prompt_id=prompt.id,
                            provider_key=panel.key,
                            model=model,
                            mode=mode,
                            status="failed" if failed_lane else "completed",
                            response_text=answer_text,
                            citations_json=citations_json,
                            usage_json=usage_json,
                            error_message=(
                                "provider unavailable" if failed_lane else None
                            ),
                        )
                        session.add(answer)
                        await session.flush()
                        if not failed_lane:
                            session.add(
                                AnswerAnnotation(
                                    answer_id=answer.id,
                                    annotation_json={
                                        "_annotation_version": ANNOTATION_VERSION,
                                        "_answer_sha256": hashlib.sha256(
                                            answer_text.encode("utf-8")
                                        ).hexdigest(),
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
            expected_cells = await _expected_corpus_cells(run_id, critic_rows)
            self.assertEqual(len(critic_rows), 81)
            self.assertEqual(len(expected_cells), 81)
            admission = build_panel_metric_coverage_admission(
                expected_cells=expected_cells,
                observed_rows=critic_rows,
            )
            self.assertTrue(admission["allowed"])
            self.assertEqual(
                admission["unavailable_provider_count_by_mode"],
                {"memory": 1},
            )
            manifest = _final_corpus_manifest(
                critic_rows,
                expected_cells=expected_cells,
            )
            self.assertTrue(manifest["structural_complete"])
            self.assertTrue(manifest["evidentiary_complete"])
            self.assertTrue(manifest["complete"])
            self.assertEqual(len(manifest["unavailable_cells"]), 9)
            critic_gate = {
                "passed": True,
                "corpus_manifest": manifest,
                "panel_metric_coverage_admission": admission,
                "panel_metric_coverage_admission_sha256": admission["admission_sha256"],
                "provenance": {
                    "rows_sha256": manifest["critic_rows_sha256"],
                },
            }

            corpus = await _full_answer_context(
                run_id,
                critic_gate=critic_gate,
                critic_rows=critic_rows,
                expected_corpus_cells=expected_cells,
            )
            selected, selection_manifest = _select_final_answer_context(
                corpus["answers"],
                corpus_manifest=corpus["manifest"],
            )
            self.assertEqual(len(selected), 81)
            self.assertEqual(selection_manifest["selected_metadata_only_count"], 9)
            failed = [item for item in selected if item["status"] == "failed"]
            self.assertEqual(len(failed), 9)
            for item in failed:
                self.assertEqual(item["context_access"], "metadata_only")
                self.assertEqual(
                    item["metric_limitation"],
                    "terminal_panel_failure",
                )
                self.assertNotIn("answer_text", item)
                self.assertNotIn("annotation", item)
                self.assertNotIn("citations", item)
                self.assertGreater(item["failure"]["error_utf8_bytes"], 0)
        finally:
            async with SessionLocal() as session:
                await session.execute(delete(Run).where(Run.id == run_id))
                await session.commit()


class ContentExtractionTests(unittest.TestCase):
    def test_jsonld_keeps_all_types_and_exposes_every_parse_failure(self) -> None:
        type_names = [f"CustomType{index:02d}" for index in range(45)]
        valid = json.dumps(
            {
                "@context": "https://schema.org",
                "@graph": [{"@type": value} for value in type_names],
            },
            ensure_ascii=False,
        )
        html = f"""
        <html><head>
          <script type="Application/LD+JSON; charset=utf-8">{valid}</script>
          <script type="application/ld+json">
            {{"@context":"https://schema.org","@type":"Broken",
          </script>
          <script>window.app = true;</script>
        </head><body><main><h1>Example</h1></main></body></html>
        """

        signals = extract_text_signals(html, "text/html")

        self.assertEqual(signals["script_count"], 3)
        self.assertEqual(signals["structured_data_types"], type_names)
        self.assertEqual(len(signals["structured_data_types"]), 45)
        self.assertIs(signals["structured_data_complete"], False)
        self.assertEqual(signals["jsonld"]["script_count"], 2)
        self.assertEqual(signals["jsonld"]["parsed_count"], 1)
        self.assertEqual(signals["jsonld"]["failed_count"], 1)
        self.assertEqual(signals["jsonld"]["state"], "partial")
        self.assertEqual(len(signals["jsonld"]["errors"]), 1)
        error = signals["jsonld"]["errors"][0]
        self.assertEqual(error["script_index"], 1)
        self.assertEqual(error["error_type"], "json_decode_error")
        self.assertIn("property name", error["message"])
        self.assertGreater(error["line"], 0)
        self.assertGreater(error["column"], 0)
        self.assertGreater(error["char_offset"], 0)

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
    def test_two_character_real_entity_alias_is_supported(self) -> None:
        entries = _entity_alias_entries(
            {
                "canonical_name": "VK",
                "aliases": ["ВКонтакте"],
                "mention_policy": "standalone",
            },
            {"brand_name": "VK"},
            excluded_aliases=[],
        )

        self.assertIn(("VK", "standalone"), entries)
        self.assertTrue(
            _evidence_contains_complete_alias(
                "VK развивает рекламную платформу.",
                entries,
            )
        )

    def test_unknown_mentioned_position_is_not_a_top3_failure(self) -> None:
        def row(*, mentioned: bool, position: int | None, role: str) -> dict[str, Any]:
            return {
                "mode": "web",
                "role": "unbranded_discovery",
                "provider_key": "test",
                "intent_class": "I",
                "status": "completed",
                "answer_text": "Непустой ответ",
                "metric_eligible": True,
                "annotation": {
                    "valid": True,
                    "target_mentioned": mentioned,
                    "target_position": position,
                    "target_role": role,
                },
            }

        result = _visibility_slice(
            [
                row(mentioned=True, position=None, role="mentioned"),
                row(mentioned=False, position=None, role="absent"),
                row(mentioned=True, position=2, role="recommended"),
            ],
            mode="web",
        )

        self.assertEqual(result["valid_answers"], 3)
        self.assertEqual(result["top3_denominator"], 2)
        self.assertEqual(result["top3_count"], 1)
        self.assertEqual(result["top3_rate"], 50.0)

    def test_evidence_must_be_an_exact_contiguous_raw_substring(self) -> None:
        raw = "**Jois**\nквартиры с зелёными террасами"

        self.assertTrue(_evidence_is_literal(raw, "Jois"))
        self.assertTrue(_evidence_is_literal(raw, "квартиры с зелёными террасами"))
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
                ("- JOIS (MR Group). В продаже есть квартиры с приватными террасами."),
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
                                    "match_policy": ("requires_target_attribution"),
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
                        "Облачная платформа аналитики (бизнес-юнит Northstar Group)"
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
                item["name"] == "Orbit Cloud" and item["relationship"] == "competitor"
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
                "prompt_key": "u-1",
                "intent_class": "I",
                "role": "unbranded_discovery",
                "status": "completed",
                "answer_text": "RW+ рекомендуется как целевой бренд.",
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
                "answer_text": "В ответе перечислены другие бренды.",
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
                "prompt_key": "u-2",
                "intent_class": "E",
                "role": "unbranded_discovery",
                "status": "completed",
                "answer_text": "RW+ рекомендуется как целевой бренд.",
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
        self.assertIsNone(metrics["paired_web_lift"]["parent"]["score_lift"])
        self.assertIsNone(metrics["knowledge_gap"])
        self.assertEqual(metrics["quality"]["state"], "limited")
        self.assertEqual(metrics["quality"]["coverage_rate"], 50.0)

    def test_paired_lift_is_limited_when_one_planned_half_failed(self) -> None:
        valid = {
            "valid": True,
            "target_mentioned": False,
            "target_position": None,
            "target_role": "absent",
            "sentiment": "unknown",
            "entity_mentions": [],
        }
        rows = [
            {
                "answer_id": 1,
                "mode": "web",
                "provider_key": "openai",
                "prompt_id": 1,
                "prompt_key": "pair-1",
                "intent_class": "I",
                "role": "unbranded_discovery",
                "status": "completed",
                "annotation": dict(valid),
            },
            {
                "answer_id": 2,
                "mode": "memory",
                "provider_key": "openai",
                "prompt_id": 1,
                "prompt_key": "pair-1",
                "intent_class": "I",
                "role": "unbranded_discovery",
                "status": "completed",
                "annotation": dict(valid),
            },
            {
                "answer_id": 3,
                "mode": "web",
                "provider_key": "openai",
                "prompt_id": 2,
                "prompt_key": "pair-2",
                "intent_class": "E",
                "role": "unbranded_discovery",
                "status": "completed",
                "annotation": dict(valid),
            },
            {
                "answer_id": 4,
                "mode": "memory",
                "provider_key": "openai",
                "prompt_id": 2,
                "prompt_key": "pair-2",
                "intent_class": "E",
                "role": "unbranded_discovery",
                "status": "failed",
                "annotation": {},
            },
        ]
        metrics = _compute_metrics(rows, {"brand_name": "Цель"}, {"entities": []})
        paired = metrics["paired_web_lift"]
        self.assertEqual(paired["n_pairs"], 1)
        self.assertEqual(paired["expected_pairs"], 2)
        self.assertEqual(paired["completed_pairs"], 1)
        self.assertEqual(paired["missing_pairs"], 1)
        self.assertEqual(paired["coverage_rate"], 50.0)
        self.assertEqual(paired["data_state"], "limited")
        self.assertEqual(
            paired["limitation_reason"],
            "incomplete_paired_coverage",
        )
        self.assertEqual(paired["parent"]["web"]["data_state"], "limited")
        self.assertEqual(paired["parent"]["memory"]["data_state"], "limited")

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
                "answer_text": "RW+ рекомендуется как целевой бренд.",
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
                "answer_text": "В ответе перечислены другие бренды.",
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
                "answer_text": "RW+ рекомендуется как целевой бренд.",
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
                "answer_text": "В ответе перечислены другие бренды.",
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
                "answer_text": "RW+ рекомендуется как целевой бренд.",
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
        intent_i = next(item for item in metrics["intents"] if item["intent"] == "I")
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
                row.get("relationship") == "portfolio" for row in metrics["competitors"]
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
                    "answer_text": "Цель рекомендуется для этого сценария.",
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
                    "answer_text": "Ответ описывает другие решения.",
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
            [item["name"] for item in metrics["brand_knowledge"]["web"]["facets"]],
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
                    "web_attestation_reason": (LEGACY_MEMORY_OBSERVATION_REASON),
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
        self.assertIsNone(
            metrics["paired_web_lift"]["portfolio"]["score_lift"],
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
                        "### Realweb\n- Сильная сторона: Okkam предлагает аналитику."
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
            competitor_dooh is None or competitor_dooh["attributed_to_target"] is False
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
            "Omni360 — независимая DSP-платформа. Programmatic использует DSP и DMP."
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
        answer_text = "Okkam предлагает DOOH; Realweb предлагает аналитику."
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
        answer_text = "Example предлагает разработку стратегий и ведение кампаний."

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
            "Okkam предлагает разработку стратегий, а Example упомянут для сравнения."
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
            ("Realweb (Риалвеб)\nВ чём сила: медиазакупка и DOOH."),
            (
                "Realweb. Сильная сторона: универсальное агентство. "
                "Хорошо сочетает performance и DOOH."
            ),
            ("Realweb (Риалвеб)\nАналитика и технологии: DOOH-дашборды."),
            ("Realweb\nАналитика и ПО: собственная DOOH-платформа."),
            (
                "Realweb\n"
                "Сильная сторона: performance\n"
                "Компетенции: цифровая наружная реклама"
            ),
            (
                "### Realweb\nОписание: "
                + ("подробный подтверждённый контекст агентства " * 36)
                + "\n- Сильная сторона: DOOH"
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
                        "Realweb — предлагает планирование и закупку programmatic DOOH."
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
        self.assertFalse(false_reconciled["entity_mentions"][0]["attributed_to_target"])

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

        self.assertIsNone(metrics["portfolio_visibility"]["web"]["mention_count"])
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

    def test_profile_service_scope_survives_empty_observed_catalog_and_requires_attribution(
        self,
    ) -> None:
        profile = _profile_with_offer_contract(
            {
                "brand_name": "Realweb",
                "brand_aliases": [],
                "entity_scope": [],
                "products": [],
                "customer_jobs": ["Запустить наружную рекламу"],
            },
            domain="realweb.example",
            offer_name="Custom tattoos",
            offer_kind="service",
        )
        rows = [
            {
                "answer_id": 701,
                "mode": "web",
                "provider_key": "openai",
                "prompt_id": 1,
                "prompt_key": "u-1",
                "intent_class": "I",
                "role": "unbranded_discovery",
                "status": "completed",
                "answer_text": "Custom tattoos are available in many studios.",
                "annotation": {
                    "valid": True,
                    "target_mentioned": False,
                    "target_position": None,
                    "target_role": "absent",
                    "sentiment": "unknown",
                    "entity_mentions": [
                        {
                            "canonical_name": "Custom tattoos",
                            "position": 1,
                            "role": "mentioned",
                            "attributed_to_target": False,
                            "evidence": "Custom tattoos are available in many studios.",
                        }
                    ],
                },
            }
        ]

        metrics = _compute_metrics(
            rows,
            profile,
            {"target_aliases": [], "entities": [], "uncertainties": []},
        )

        self.assertEqual(metrics["portfolio_scope"]["state"], "complete")
        self.assertEqual(metrics["portfolio_scope"]["confirmed_entities"], 1)
        self.assertEqual(metrics["portfolio_visibility"]["web"]["mention_count"], 0)
        self.assertEqual(
            _portfolio_mention_policy(
                {
                    "canonical_name": "Custom tattoos",
                    "category": "target",
                    "target_relationship": "portfolio_entity",
                    "commercially_relevant": True,
                    "mention_policy": "standalone",
                },
                profile,
            ),
            "requires_target_attribution",
        )

    def test_common_product_phrase_needs_target_attribution(self) -> None:
        source_text = "Example offers running shoes for everyday training."
        source = SourceUnit.from_text(
            source_unit_id="https://example.com/shoes:000000",
            source_url="https://example.com/shoes",
            text=source_text,
        )
        offer_catalog = build_offer_catalog(
            client_domain="example.com",
            client_aliases=["Example"],
            source_units=[source],
            candidates=[
                {
                    "canonical_name": "running shoes",
                    "aliases": [],
                    "kind": "product",
                    "source_url": source.source_url,
                    "evidence_excerpt": source_text,
                    "source_unit_id": source.source_unit_id,
                    "source_sha256": source.source_sha256,
                    "confidence": 0.95,
                    "user_jobs": ["Buy running shoes"],
                    "commercially_relevant": True,
                }
            ],
        )
        profile = {
            "brand_name": "Example",
            "brand_aliases": [],
            "products": ["running shoes"],
            "entity_scope": [],
            "offer_catalog": offer_catalog.as_dict(),
        }

        def row(
            answer_id: int,
            answer_text: str,
            *,
            target_mentioned: bool,
            attributed: bool,
        ) -> dict[str, Any]:
            return {
                "answer_id": answer_id,
                "mode": "web",
                "provider_key": "openai",
                "prompt_id": answer_id,
                "prompt_key": f"u-{answer_id}",
                "intent_class": "I",
                "role": "unbranded_discovery",
                "status": "completed",
                "answer_text": answer_text,
                "annotation": {
                    "valid": True,
                    "target_mentioned": target_mentioned,
                    "target_position": 1 if target_mentioned else None,
                    "target_role": "mentioned" if target_mentioned else "absent",
                    "sentiment": "neutral",
                    "entity_mentions": [
                        {
                            "canonical_name": "running shoes",
                            "position": 1,
                            "role": "mentioned",
                            "attributed_to_target": attributed,
                            "evidence": answer_text,
                        }
                    ],
                },
            }

        unbranded = _compute_metrics(
            [
                row(
                    801,
                    "Running shoes are available from many brands.",
                    target_mentioned=False,
                    attributed=False,
                )
            ],
            profile,
            {"target_aliases": [], "entities": [], "uncertainties": []},
        )
        self.assertEqual(
            unbranded["portfolio_visibility"]["web"]["mention_count"],
            0,
        )

        attributed = _compute_metrics(
            [
                row(
                    802,
                    "Example offers running shoes for everyday training.",
                    target_mentioned=True,
                    attributed=True,
                )
            ],
            profile,
            {"target_aliases": [], "entities": [], "uncertainties": []},
        )
        self.assertEqual(
            attributed["portfolio_visibility"]["web"]["mention_count"],
            1,
        )

    def test_prefix_overlapping_offers_keep_independent_profile_scope(self) -> None:
        names = ["Campaign 360", "Campaign 360 Enterprise Analytics"]
        sources: list[SourceUnit] = []
        candidates: list[dict[str, Any]] = []
        for index, name in enumerate(names):
            source_text = f"Example offers product {name} for campaign analysis."
            source_url = f"https://example.com/product-{index}"
            source = SourceUnit.from_text(
                source_unit_id=f"{source_url}:000000",
                source_url=source_url,
                text=source_text,
            )
            sources.append(source)
            candidates.append(
                {
                    "canonical_name": name,
                    "aliases": [],
                    "kind": "product",
                    "source_url": source.source_url,
                    "evidence_excerpt": source_text,
                    "source_unit_id": source.source_unit_id,
                    "source_sha256": source.source_sha256,
                    "confidence": 0.95,
                    "user_jobs": [f"Use {name}"],
                    "commercially_relevant": True,
                }
            )
        offer_catalog = build_offer_catalog(
            client_domain="example.com",
            client_aliases=["Example"],
            source_units=sources,
            candidates=candidates,
        )
        profile = {
            "brand_name": "Example",
            "brand_aliases": [],
            "products": names,
            "entity_scope": [],
            "offer_catalog": offer_catalog.as_dict(),
        }
        observed = {
            "target_aliases": ["Example"],
            "entities": [
                {
                    "canonical_name": "Campaign 360",
                    "aliases": [],
                    "category": "target",
                    "target_relationship": "portfolio_entity",
                    "commercially_relevant": True,
                    "mention_policy": "standalone",
                    "evidence": "Campaign 360",
                }
            ],
            "uncertainties": [],
        }

        scoped = _scope_entity_catalog_to_profile(observed, profile)
        scoped_names = [
            item["canonical_name"]
            for item in scoped["entities"]
            if item.get("category") == "target"
        ]
        self.assertEqual(scoped_names, names)

        metrics = _compute_metrics([], profile, observed)
        self.assertEqual(metrics["portfolio_scope"]["state"], "complete")
        self.assertEqual(metrics["portfolio_scope"]["confirmed_entities"], 2)
        self.assertEqual(metrics["portfolio_scope"]["rejected_entities"], 0)

    def test_primary_brand_can_also_be_the_profile_product(self) -> None:
        profile = _profile_with_offer_contract(
            {
                "brand_name": "Spotify",
                "brand_aliases": [],
                "entity_scope": [],
                "products": [],
                "customer_jobs": ["Listen to music"],
            },
            domain="spotify.example",
            offer_name="Spotify",
            offer_kind="product",
        )
        observed = {
            "target_aliases": ["Spotify"],
            "entities": [
                {
                    "canonical_name": "Spotify",
                    "aliases": [],
                    "category": "target",
                    "target_relationship": "exact_target",
                    "commercially_relevant": True,
                    "mention_policy": "standalone",
                    "evidence": "Spotify",
                }
            ],
            "uncertainties": [],
        }

        scoped = _scope_entity_catalog_to_profile(observed, profile)
        self.assertEqual(len(scoped["entities"]), 1)
        self.assertTrue(scoped["entities"][0]["_also_exact_target"])
        self.assertTrue(scoped["entities"][0]["_profile_membership_confirmed"])

        metrics = _compute_metrics([], profile, observed)
        self.assertEqual(metrics["portfolio_scope"]["state"], "complete")
        self.assertEqual(metrics["portfolio_scope"]["confirmed_entities"], 1)
        self.assertEqual(metrics["portfolio_scope"]["rejected_entities"], 0)

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
            all(item["answer_count"] == 1 for item in portfolio["mentioned_entities"])
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
        self.assertEqual(report["headline_emphasis"], [])

    def test_competitor_series_retains_every_observed_entity(self) -> None:
        competitor_names = [f"Конкурент {index:02d}" for index in range(20)]
        answer_text = ", ".join(competitor_names)
        rows = [
            {
                "answer_id": 501,
                "mode": "web",
                "provider_key": "openai",
                "prompt_id": 51,
                "prompt_key": "all-competitors",
                "intent_class": "E",
                "role": "unbranded_discovery",
                "status": "completed",
                "answer_text": answer_text,
                "annotation": {
                    "valid": True,
                    "target_mentioned": False,
                    "target_position": None,
                    "target_role": "absent",
                    "sentiment": "unknown",
                    "entity_mentions": [
                        {
                            "canonical_name": name,
                            "position": index + 1,
                            "role": "mentioned",
                            "attributed_to_target": False,
                            "evidence": name,
                        }
                        for index, name in enumerate(competitor_names)
                    ],
                },
            }
        ]
        profile = {
            "brand_name": "Целевой бренд",
            "brand_aliases": [],
            "products": [],
            "entity_scope": [],
        }
        catalog = {
            "target_aliases": ["Целевой бренд"],
            "entities": [
                {
                    "canonical_name": name,
                    "aliases": [],
                    "category": "competitor",
                    "target_relationship": "competitor",
                    "commercially_relevant": True,
                    "mention_policy": "standalone",
                }
                for name in competitor_names
            ],
        }

        metrics = _compute_metrics(rows, profile, catalog)

        self.assertEqual(len(metrics["competitors"]), 21)
        self.assertEqual(
            {item["name"] for item in metrics["competitors"]},
            {"Целевой бренд", *competitor_names},
        )
        self.assertEqual(
            metrics["competitor_series_contract"]["canonical_rows_omitted"],
            0,
        )
        self.assertEqual(
            metrics["competitor_series_contract"]["total_rows"],
            21,
        )


class OfferIdentityPipelineIntegrationTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _profile() -> dict[str, Any]:
        definitions = (
            (
                "Centra",
                (),
                "service",
                "Example развивает сервис Centra для покупки рекламы.",
            ),
            (
                "Campaign 360",
                ("campaign",),
                "product",
                "Example предлагает платформу Campaign 360 для планирования.",
            ),
            (
                "eCommerce",
                (),
                "service",
                "Example оказывает услуги eCommerce для магазинов.",
            ),
        )
        sources: list[SourceUnit] = []
        candidates: list[dict[str, Any]] = []
        for index, (name, aliases, kind, source_text) in enumerate(definitions):
            source = SourceUnit.from_text(
                source_unit_id=f"https://example.com/{index}:000000",
                source_url=f"https://example.com/{index}",
                text=source_text,
            )
            sources.append(source)
            candidates.append(
                {
                    "canonical_name": name,
                    "aliases": list(aliases),
                    "kind": kind,
                    "source_url": source.source_url,
                    "evidence_excerpt": source_text,
                    "source_unit_id": source.source_unit_id,
                    "source_sha256": source.source_sha256,
                    "confidence": 0.95,
                    "user_jobs": ["Выбрать поставщика"],
                    "commercially_relevant": True,
                }
            )
        catalog = build_offer_catalog(
            client_domain="example.com",
            client_aliases=["Example"],
            source_units=sources,
            candidates=candidates,
        )
        return {
            "brand_name": "Example",
            "brand_aliases": [],
            "products": list(catalog.legacy_product_strings()),
            "entity_scope": [],
            "offer_catalog": catalog.as_dict(),
        }

    @staticmethod
    def _batch_from_call(
        user_payload: dict[str, Any],
        *,
        decisions: dict[str, OfferIdentityDecision] | None = None,
    ) -> dict[str, Any]:
        role = OfferIdentityModelRole(user_payload["request_contract"]["role"])
        decisions = decisions or {
            "Centra": OfferIdentityDecision.NAMED_OFFERING,
            "Campaign 360": OfferIdentityDecision.NAMED_OFFERING,
            "campaign": OfferIdentityDecision.GENERIC_CATEGORY,
            "eCommerce": OfferIdentityDecision.GENERIC_CATEGORY,
        }
        reason_by_decision = {
            OfferIdentityDecision.NAMED_OFFERING: (
                OfferIdentityReasonCode.EXPLICIT_NAMED_IDENTITY
            ),
            OfferIdentityDecision.GENERIC_CATEGORY: (
                OfferIdentityReasonCode.DESCRIPTIVE_OFFERING_PHRASE
            ),
            OfferIdentityDecision.AMBIGUOUS: (
                OfferIdentityReasonCode.INSUFFICIENT_IDENTITY_EVIDENCE
            ),
        }
        contract = user_payload["contract"]
        return {
            "version": OFFER_IDENTITY_MODEL_BATCH_VERSION,
            "role": role.value,
            "input_digest": contract["input_digest"],
            "policy_digest": contract["policy_digest"],
            "catalog_digest": contract["catalog_digest"],
            "request_contract_digest": user_payload["request_contract"][
                "request_contract_digest"
            ],
            "subject_count": contract["subject_count"],
            "decisions": [
                {
                    "version": OFFER_IDENTITY_MODEL_DECISION_VERSION,
                    "role": role.value,
                    "subject_id": subject["subject_id"],
                    "input_digest": contract["input_digest"],
                    "policy_digest": contract["policy_digest"],
                    "catalog_digest": contract["catalog_digest"],
                    "subject_digest": subject["subject_digest"],
                    "evidence_refs_digest": subject["evidence_refs_digest"],
                    "reviewed_evidence_ref_digests": subject[
                        "evidence_ref_digests"
                    ],
                    "decision": decisions[subject["name"]].value,
                    "reason_code": reason_by_decision[
                        decisions[subject["name"]]
                    ].value,
                    "rationale": "Решение принято по точному фрагменту сайта.",
                }
                for subject in user_payload["subjects"]
            ],
        }

    async def test_two_independent_catalog_calls_enable_only_named_offers(
        self,
    ) -> None:
        profile = self._profile()

        async def model_call(*_args: Any, **kwargs: Any) -> dict[str, Any]:
            return self._batch_from_call(kwargs["user_payload"])

        with (
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
                "app.services.analyzer._structured_artifact",
                new_callable=AsyncMock,
                side_effect=model_call,
            ) as structured,
        ):
            enriched = await _attach_offer_identity_policy("run-id", profile)

        self.assertEqual(structured.await_count, 2)
        calls_by_role = {
            call.kwargs["user_payload"]["request_contract"]["role"]: call
            for call in structured.await_args_list
        }
        self.assertEqual(set(calls_by_role), {"primary", "critic"})
        primary_payload = calls_by_role["primary"].kwargs["user_payload"]
        critic_payload = calls_by_role["critic"].kwargs["user_payload"]
        self.assertEqual(primary_payload["subjects"], critic_payload["subjects"])
        self.assertNotIn("primary_result", critic_payload)
        self.assertTrue(
            all(call.kwargs["continuable"] for call in structured.await_args_list)
        )

        scoped = {
            row["canonical_name"]: row
            for row in _profile_offer_scope_entities(enriched)
        }
        self.assertEqual(scoped["Centra"]["mention_policy"], "standalone")
        self.assertEqual(
            scoped["Campaign 360"]["mention_policy"],
            "standalone",
        )
        self.assertEqual(
            scoped["Campaign 360"]["aliases"],
            [
                {
                    "value": "campaign",
                    "match_policy": "requires_target_attribution",
                }
            ],
        )
        self.assertEqual(
            scoped["eCommerce"]["mention_policy"],
            "requires_target_attribution",
        )
        self.assertEqual(
            enriched["_offer_identity_policy"]["quality_state"],
            "complete",
        )
        completed = save_artifact.await_args_list[-1]
        self.assertEqual(completed.kwargs["status"], "completed")
        self.assertEqual(completed.kwargs["model"], None)

    async def test_provider_failure_degrades_policy_without_stopping_report(
        self,
    ) -> None:
        profile = self._profile()

        async def model_call(*_args: Any, **kwargs: Any) -> dict[str, Any]:
            payload = kwargs["user_payload"]
            if payload["request_contract"]["role"] == "primary":
                raise OpenRouterError("provider unavailable")
            return self._batch_from_call(payload)

        with (
            patch(
                "app.services.analyzer._artifact_output",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ),
            patch(
                "app.services.analyzer._structured_artifact",
                new_callable=AsyncMock,
                side_effect=model_call,
            ),
        ):
            enriched = await _attach_offer_identity_policy("run-id", profile)

        summary = enriched["_offer_identity_policy"]
        self.assertEqual(summary["quality_state"], "degraded")
        self.assertEqual(summary["standalone_count"], 0)
        self.assertIn("primary:role_error", summary["diagnostic_codes"])
        self.assertTrue(
            all(
                row["mention_policy"] == "requires_target_attribution"
                for row in _profile_offer_scope_entities(enriched)
            )
        )

    async def test_contract_invalid_completed_leaf_is_made_retryable(self) -> None:
        profile = self._profile()

        async def model_call(*_args: Any, **kwargs: Any) -> dict[str, Any]:
            payload = kwargs["user_payload"]
            batch = self._batch_from_call(payload)
            if payload["request_contract"]["role"] == "primary":
                batch["decisions"][-1] = copy.deepcopy(batch["decisions"][0])
            return batch

        with (
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
                "app.services.analyzer._structured_artifact",
                new_callable=AsyncMock,
                side_effect=model_call,
            ),
        ):
            enriched = await _attach_offer_identity_policy("run-id", profile)

        self.assertEqual(
            enriched["_offer_identity_policy"]["quality_state"],
            "degraded",
        )
        failed_leaf_writes = [
            call
            for call in save_artifact.await_args_list
            if call.kwargs.get("status") == "failed"
            and str(call.kwargs.get("artifact_key") or "").startswith(
                "offer_identity_primary_"
            )
        ]
        self.assertEqual(len(failed_leaf_writes), 1)
        self.assertTrue(
            failed_leaf_writes[0].kwargs["preserve_existing_evidence"]
        )

    def test_policy_digest_changes_annotation_contract(self) -> None:
        profile = self._profile()
        contract = build_offer_identity_contract(
            OfferCatalog.from_mapping(profile["offer_catalog"])
        )
        base_summary = {
            "version": "aiv-offer-identity-result-v1",
            "catalog_digest": contract.catalog_digest,
            "input_digest": contract.input_digest,
            "output_digest": "a" * 64,
            "standalone_names_by_offer": {},
        }
        first = {**profile, "_offer_identity_policy": base_summary}
        second = {
            **profile,
            "_offer_identity_policy": {
                **base_summary,
                "output_digest": "b" * 64,
            },
        }
        observed = {"target_aliases": ["Example"], "entities": []}
        self.assertNotEqual(
            _annotation_context_sha256(first, observed),
            _annotation_context_sha256(second, observed),
        )

    def test_named_alias_is_independent_from_generic_canonical_name(self) -> None:
        source_text = (
            "Example предлагает услугу SEO, также называемую Garpun, "
            "для продвижения."
        )
        source = SourceUnit.from_text(
            source_unit_id="https://example.com/seo:000000",
            source_url="https://example.com/seo",
            text=source_text,
        )
        offer_catalog = build_offer_catalog(
            client_domain="example.com",
            client_aliases=["Example"],
            source_units=[source],
            candidates=[
                {
                    "canonical_name": "SEO",
                    "aliases": ["Garpun"],
                    "kind": "service",
                    "source_url": source.source_url,
                    "evidence_excerpt": source_text,
                    "source_unit_id": source.source_unit_id,
                    "source_sha256": source.source_sha256,
                    "confidence": 0.95,
                    "user_jobs": ["Продвигать сайт"],
                    "commercially_relevant": True,
                }
            ],
        )
        offer = offer_catalog.accepted_offers[0]
        contract = build_offer_identity_contract(offer_catalog)
        profile = {
            "brand_name": "Example",
            "brand_aliases": [],
            "products": ["SEO"],
            "entity_scope": [],
            "offer_catalog": offer_catalog.as_dict(),
            "_offer_identity_policy": {
                "version": OFFER_IDENTITY_RESULT_VERSION,
                "catalog_digest": offer_catalog.catalog_digest,
                "input_digest": contract.input_digest,
                "output_digest": "a" * 64,
                "standalone_names_by_offer": {offer.offer_id: ["Garpun"]},
            },
        }

        scoped = _profile_offer_scope_entities(profile)

        self.assertEqual(scoped[0]["mention_policy"], "requires_target_attribution")
        self.assertEqual(
            scoped[0]["aliases"],
            [{"value": "Garpun", "match_policy": "standalone"}],
        )
        self.assertEqual(scoped[0]["_profile_standalone_match_aliases"], ["Garpun"])


class DatabaseSafetyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        await init_db()

    @staticmethod
    def _checkpoint_prompt_items() -> list[dict[str, object]]:
        intents = ("I", "E", "T", "NB", "NAV", "TR", "I", "E", "TR")
        return [
            {
                "prompt_key": f"checkpoint-{sequence}",
                "intent_class": intent,
                "role": (
                    "unbranded_discovery" if sequence <= 6 else "brand_diagnostic"
                ),
                "text": f"Сохранённый сценарий {sequence}",
                "rationale": f"Проверяет сигнал {sequence}",
                "choice_request": sequence <= 6,
            }
            for sequence, intent in enumerate(intents, start=1)
        ]

    async def _create_panel_checkpoint(
        self,
        *,
        prompt_version: str = PROMPT_SET_VERSION,
        artifact_mismatch: bool = False,
        answer_status: str = "completed",
        with_annotation: bool = False,
    ) -> tuple[str, list[dict[str, object]], list[int], int]:
        run_id = f"test-panel-checkpoint-{uuid.uuid4()}"
        items = self._checkpoint_prompt_items()
        async with SessionLocal() as session:
            session.add(
                Run(
                    id=run_id,
                    domain="example.com",
                    status=RunStatus.analyzing,
                    config_json={},
                )
            )
            prompts: list[VisibilityPrompt] = []
            for sequence, item in enumerate(items, start=1):
                prompt = VisibilityPrompt(
                    run_id=run_id,
                    prompt_key=str(item["prompt_key"]),
                    intent_class=str(item["intent_class"]),
                    role=str(item["role"]),
                    text=str(item["text"]),
                    rationale=str(item["rationale"]),
                    sequence=sequence,
                )
                session.add(prompt)
                prompts.append(prompt)
            await session.flush()
            artifact_items = copy.deepcopy(items)
            if artifact_mismatch:
                artifact_items[0]["text"] = "Сценарий не совпадает с БД"
            session.add(
                RunArtifact(
                    run_id=run_id,
                    stage_key="scenario_design",
                    artifact_key="prompt_set",
                    status="completed",
                    prompt_version=prompt_version,
                    output_json={"prompts": artifact_items},
                )
            )
            answer = ModelAnswer(
                run_id=run_id,
                prompt_id=prompts[0].id,
                provider_key="openai",
                model="test/model",
                mode="web",
                status=answer_status,
                response_text=(
                    "Сохранённый raw-ответ" if answer_status == "completed" else None
                ),
            )
            session.add(answer)
            await session.flush()
            if with_annotation:
                session.add(
                    AnswerAnnotation(
                        answer_id=answer.id,
                        annotation_json={"valid": True},
                    )
                )
            await session.commit()
            return run_id, items, [prompt.id for prompt in prompts], answer.id

    async def _create_full_panel_corpus(self) -> tuple[str, list[int]]:
        run_id = f"test-full-panel-corpus-{uuid.uuid4()}"
        items = self._checkpoint_prompt_items()
        answer_ids: list[int] = []
        async with SessionLocal() as session:
            session.add(
                Run(
                    id=run_id,
                    domain="example.com",
                    status=RunStatus.analyzing,
                    config_json={},
                )
            )
            prompts: list[VisibilityPrompt] = []
            for sequence, item in enumerate(items, start=1):
                prompt = VisibilityPrompt(
                    run_id=run_id,
                    prompt_key=str(item["prompt_key"]),
                    intent_class=str(item["intent_class"]),
                    role=str(item["role"]),
                    text=str(item["text"]),
                    rationale=str(item["rationale"]),
                    sequence=sequence,
                )
                session.add(prompt)
                prompts.append(prompt)
            await session.flush()
            session.add(
                RunArtifact(
                    run_id=run_id,
                    stage_key="scenario_design",
                    artifact_key="prompt_set",
                    status="completed",
                    prompt_version=PROMPT_SET_VERSION,
                    output_json={"prompts": copy.deepcopy(items)},
                )
            )
            cell_index = 0
            for prompt in prompts:
                for mode in ("web", "memory"):
                    for panel in panel_models():
                        model = panel.model if mode == "web" else panel.memory_model
                        if model is None:
                            continue
                        cell_index += 1
                        response_text = (
                            f"Ответ {prompt.sequence}/{panel.key}/{mode}: хвост ✓"
                        )
                        answer = ModelAnswer(
                            run_id=run_id,
                            prompt_id=prompt.id,
                            provider_key=panel.key,
                            model=model,
                            mode=mode,
                            status="completed",
                            response_text=response_text,
                            citations_json=[
                                {
                                    "url": f"https://source.example/{cell_index}",
                                    "title": f"Источник {cell_index}",
                                }
                            ],
                            usage_json={
                                "prompt_tokens": cell_index,
                                "completion_tokens": cell_index + 1,
                                "total_tokens": cell_index * 2 + 1,
                            },
                        )
                        session.add(answer)
                        await session.flush()
                        answer_ids.append(answer.id)
            await session.commit()
        self.assertEqual(len(answer_ids), PANEL_CORPUS_EXPECTED_CELL_COUNT)
        return run_id, answer_ids

    @staticmethod
    async def _delete_run(run_id: str) -> None:
        async with SessionLocal() as session:
            await session.execute(delete(Run).where(Run.id == run_id))
            await session.commit()

    async def _single_panel_fixture(
        self,
    ) -> tuple[str, VisibilityPrompt, PanelModel]:
        run_id = f"test-panel-limit-{uuid.uuid4()}"
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
            await session.commit()
        return (
            run_id,
            prompt,
            PanelModel(
                key="openai",
                label="ChatGPT",
                model="test/model",
                memory_model="test/model",
            ),
        )

    @staticmethod
    def _panel_result(
        *,
        prompt_text: str,
        text_value: str,
        limited: bool,
        prompt_tokens: int,
        completion_tokens: int,
        output_complete: bool | None = None,
    ) -> SimpleNamespace:
        usage, citations = _attested_panel_usage(
            prompt_text=prompt_text,
            mode="web",
            provider_key="openai",
            model="test/model",
        )
        usage = dict(usage)
        usage.update(
            {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
        )
        usage["_aiv_transport"] = {
            "output_limited": limited,
            "output_complete": (
                not limited if output_complete is None else output_complete
            ),
        }
        return SimpleNamespace(
            text=text_value,
            citations=citations or [],
            usage=usage,
            request_policy=usage["_aiv_request_policy"],
            web_attestation=usage["_aiv_web_attestation"],
        )

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
                        "brand_diagnostic" if sequence > 6 else "unbranded_discovery"
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
                    (
                        await session.execute(
                            select(ModelAnswer)
                            .where(ModelAnswer.run_id == run_id)
                            .order_by(ModelAnswer.id)
                        )
                    )
                    .scalars()
                    .first()
                )
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
                    (
                        await session.execute(
                            select(ModelAnswer)
                            .where(ModelAnswer.run_id == run_id)
                            .order_by(ModelAnswer.id)
                        )
                    )
                    .scalars()
                    .first()
                )
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

    async def test_site_context_uses_only_manifest_selection_with_exact_lineage(
        self,
    ) -> None:
        run_id = f"test-context-{uuid.uuid4()}"
        stored_pages: list[SitePage] = []
        async with SessionLocal() as session:
            session.add(
                Run(
                    id=run_id,
                    domain="example.com",
                    status=RunStatus.analyzing,
                    config_json={},
                )
            )
            for index in range(11):
                url = (
                    "https://example.com/"
                    if index == 0
                    else f"https://example.com/page-{index}"
                )
                main_text = f"Полный текст страницы {index}. " + ("x" * 100)
                page = SitePage(
                    run_id=run_id,
                    url=url,
                    page_kind="home" if index == 0 else "product",
                    http_status=200,
                    text_length=len(main_text),
                    main_text=main_text,
                    content_signals={
                        "_body_truncated": False,
                        "_body_read_policy": crawler._body_read_policy(
                            crawler.ProbeResult()
                        ),
                        "_source_body_sha256": f"source-body-{index}",
                    },
                )
                session.add(page)
                stored_pages.append(page)
            await session.flush()

            selected = [
                (page.url, page.page_kind)
                for page in stored_pages[: crawler.AUDIT_PAGE_DEFAULT]
            ]
            stored_by_url = {page.url: page for page in stored_pages}
            receipt = crawler._site_page_receipt(selected, stored_by_url)
            manifest_input = crawler._site_page_manifest_input("example.com")
            relevance_receipt = crawler._selection_relevance_receipt(
                homepage_url=selected[0][0],
                candidates=[
                    crawler._candidate_evidence_record(
                        page.url,
                        source="test_fixture",
                        anchor_text=(
                            "Product service book demo"
                            if index < crawler.AUDIT_PAGE_DEFAULT
                            else "Product service"
                        ),
                    )
                    for index, page in enumerate(stored_pages[1:], start=1)
                ],
                target=crawler.AUDIT_PAGE_DEFAULT,
                proposed=selected,
                attempts=[
                    crawler._candidate_attempt(page, outcome="usable")
                    for page in selected
                ],
                selected=selected,
            )
            session.add(
                RunArtifact(
                    run_id=run_id,
                    stage_key="site_discovery",
                    artifact_key=crawler.SITE_PAGE_MANIFEST_KEY,
                    status="completed",
                    prompt_version=crawler.SITE_PAGE_MANIFEST_VERSION,
                    input_json=manifest_input,
                    output_json={
                        "pages": crawler._selected_page_records(selected),
                        "expected_page_count": len(selected),
                        "selected_count": len(selected),
                        "selected_pages_sha256": crawler._selected_pages_sha256(
                            selected
                        ),
                        "discovered_candidate_count": len(stored_pages),
                        "discovered_count": len(stored_pages),
                        "page_scope": crawler.PAGE_SCOPE,
                        "selection_policy": manifest_input["selection_policy"],
                        "selection_exhausted": False,
                        "verified_exhaustion": False,
                        "legacy_snapshot": False,
                        "discovery_state": "complete",
                        "coverage_state": "bounded",
                        "site_page_receipt": receipt,
                        "commercial_relevance_receipt": (
                            crawler._selection_relevance_projection(relevance_receipt)
                        ),
                    },
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
            selected_urls = [url for url, _kind in selected]
            self.assertEqual(len(context["pages"]), crawler.AUDIT_PAGE_DEFAULT)
            self.assertEqual(
                [page["url"] for page in context["pages"]],
                selected_urls,
            )
            self.assertEqual(
                [page["url"] for page in context["selected_pages_manifest"]["pages"]],
                selected_urls,
            )
            self.assertEqual(
                [
                    (page["source_unit_id"], page["source_sha256"])
                    for page in context["pages"]
                ],
                [(page["url"], page["content_sha256"]) for page in receipt["pages"]],
            )
            self.assertTrue(
                {
                    page.url for page in stored_pages[crawler.AUDIT_PAGE_DEFAULT :]
                }.isdisjoint({page["url"] for page in context["pages"]})
            )
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
            (
                "stale-model",
                "completed",
                "Example указан снова.",
                "test/model",
                "stale_model",
            ),
            (
                "stale-context",
                "completed",
                "Example назван.",
                "test/model",
                "stale_context",
            ),
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
                    response_text=raw,
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
                annotation_model = "old/model" if stale_kind == "stale_model" else model
                annotation_context = (
                    "old-context" if stale_kind == "stale_context" else context_sha256
                )
                annotation_raw = "Старый raw." if stale_kind == "stale_raw" else raw
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
            self.assertTrue(ANNOTATION_VERSION.endswith("annotations-v21"))
            self.assertTrue(METRICS_VERSION.endswith("metrics-v23"))
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
                        select(ModelAnswer).where(ModelAnswer.run_id == run_id)
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

    async def test_panel_output_limit_is_persisted_as_one_atomic_prefix(
        self,
    ) -> None:
        run_id, prompt, panel = await self._single_panel_fixture()
        partial = self._panel_result(
            prompt_text=prompt.text,
            text_value="Полный сохранённый префикс до лимита провайдера.",
            limited=True,
            prompt_tokens=101,
            completion_tokens=31_997,
        )
        try:
            with (
                patch(
                    "app.services.analyzer.panel_models",
                    return_value=(panel,),
                ),
                patch(
                    "app.services.analyzer.chat",
                    new_callable=AsyncMock,
                    return_value=partial,
                ) as panel_chat,
                patch(
                    "app.services.analyzer.update_progress",
                    new_callable=AsyncMock,
                ),
            ):
                await _run_panel(
                    run_id,
                    [prompt],
                    mode="web",
                    start_percent=40,
                    end_percent=60,
                )

            panel_chat.assert_awaited_once()
            request = panel_chat.await_args.kwargs
            self.assertEqual(
                request["output_token_policy"],
                OutputTokenPolicy.MODEL_MAX,
            )
            self.assertTrue(request["accept_output_limited"])
            self.assertFalse(request["retry_response_contract_errors"])
            self.assertFalse(request["retry_transport_errors"])
            self.assertNotIn("max_tokens", request)
            self.assertNotIn("max_completion_tokens", request)
            async with SessionLocal() as session:
                answer = (
                    await session.execute(
                        select(ModelAnswer).where(ModelAnswer.run_id == run_id)
                    )
                ).scalar_one()
                audit_rows = list(
                    (
                        await session.execute(
                            select(RunArtifact)
                            .where(
                                RunArtifact.run_id == run_id,
                                RunArtifact.prompt_version
                                == PANEL_ATTEMPT_AUDIT_VERSION,
                            )
                            .order_by(RunArtifact.id)
                        )
                    )
                    .scalars()
                    .all()
                )
            self.assertEqual(answer.status, "completed")
            self.assertEqual(answer.response_text, partial.text)
            self.assertEqual(len(audit_rows), 1)
            audit = audit_rows[0]
            self.assertEqual(audit.input_json["max_tokens"], None)
            self.assertEqual(
                audit.input_json["output_policy"],
                PANEL_OUTPUT_POLICY,
            )
            self.assertEqual(audit.status, "completed")
            self.assertEqual(audit.raw_text, partial.text)
            self.assertIsNone(audit.output_json["error_type"])
            self.assertTrue(audit.output_json["transport"]["output_limited"])
            self.assertFalse(audit.output_json["transport"]["output_complete"])
            self.assertEqual(
                audit.output_json["response_text_sha256"],
                hashlib.sha256(partial.text.encode("utf-8")).hexdigest(),
            )
            provenance = answer.usage_json["_aiv_panel_contract"]
            self.assertEqual(provenance["version"], PANEL_CONTRACT_VERSION)
            self.assertEqual(provenance["output_policy"], PANEL_OUTPUT_POLICY)
            self.assertEqual(
                provenance["observation_completeness"],
                "provider_limited_prefix",
            )
            self.assertEqual(
                provenance["request_sha256"],
                _panel_request_sha256(
                    prompt_text=prompt.text,
                    mode="web",
                    provider_key="openai",
                    model="test/model",
                ),
            )
            self.assertEqual(
                provenance["raw_response_sha256"],
                hashlib.sha256(partial.text.encode("utf-8")).hexdigest(),
            )
            self.assertEqual(provenance["raw_response_chars"], len(partial.text))
            self.assertNotIn("_aiv_panel_retry", answer.usage_json)
            self.assertEqual(
                _panel_answer_attestation(
                    answer,
                    prompt_text=prompt.text,
                ),
                (True, "verified"),
            )
        finally:
            async with SessionLocal() as session:
                await session.execute(delete(Run).where(Run.id == run_id))
                await session.commit()

    async def test_unexpected_panel_output_limit_exception_fails_one_call(
        self,
    ) -> None:
        run_id, prompt, panel = await self._single_panel_fixture()
        partial = self._panel_result(
            prompt_text=prompt.text,
            text_value="Префикс, после которого transport ошибочно поднял исключение.",
            limited=True,
            prompt_tokens=87,
            completion_tokens=31_997,
        )
        partial.citations = [
            {"url": "https://source.example/second", "title": "Second"}
        ]
        failure = OpenRouterOutputLimitError(
            "unexpected output limit exception",
            result=partial,
        )
        try:
            with (
                patch(
                    "app.services.analyzer.panel_models",
                    return_value=(panel,),
                ),
                patch(
                    "app.services.analyzer.chat",
                    new_callable=AsyncMock,
                    side_effect=failure,
                ) as panel_chat,
                patch(
                    "app.services.analyzer.update_progress",
                    new_callable=AsyncMock,
                ),
                self.assertRaisesRegex(
                    OpenRouterError,
                    "Too few successful web panel responses",
                ),
            ):
                await _run_panel(
                    run_id,
                    [prompt],
                    mode="web",
                    start_percent=40,
                    end_percent=60,
                )

            panel_chat.assert_awaited_once()
            async with SessionLocal() as session:
                answer = (
                    await session.execute(
                        select(ModelAnswer).where(ModelAnswer.run_id == run_id)
                    )
                ).scalar_one()
            self.assertEqual(answer.status, "failed")
            self.assertEqual(answer.response_text, partial.text)
            self.assertEqual(
                answer.citations_json,
                partial.citations,
            )
            self.assertIn("unexpected output limit", answer.error_message)
            self.assertTrue(answer.usage_json["_aiv_transport"]["output_limited"])
            provenance = answer.usage_json["_aiv_panel_contract"]
            self.assertEqual(provenance["version"], PANEL_CONTRACT_VERSION)
            self.assertEqual(provenance["output_policy"], PANEL_OUTPUT_POLICY)
            self.assertEqual(
                provenance["request_sha256"],
                _panel_request_sha256(
                    prompt_text=prompt.text,
                    mode="web",
                    provider_key="openai",
                    model="test/model",
                ),
            )
            self.assertNotIn("_aiv_panel_retry", answer.usage_json)
            async with SessionLocal() as session:
                audit_rows = list(
                    (
                        await session.execute(
                            select(RunArtifact).where(
                                RunArtifact.run_id == run_id,
                                RunArtifact.prompt_version
                                == PANEL_ATTEMPT_AUDIT_VERSION,
                            )
                        )
                    )
                    .scalars()
                    .all()
                )
            self.assertEqual(len(audit_rows), 1)
            self.assertEqual(audit_rows[0].status, "failed")
            self.assertEqual(audit_rows[0].raw_text, partial.text)
            self.assertEqual(
                audit_rows[0].output_json["error_type"],
                "OpenRouterOutputLimitError",
            )
        finally:
            async with SessionLocal() as session:
                await session.execute(delete(Run).where(Run.id == run_id))
                await session.commit()

    async def test_cancelled_atomic_call_keeps_reservation_and_imported_raw(
        self,
    ) -> None:
        run_id, prompt, panel = await self._single_panel_fixture()
        historical_raw = "Исторический failed raw до новой системы аудита."
        historical_citations = [{"url": "https://old.example/source"}]
        historical_usage = {"prompt_tokens": 11, "completion_tokens": 19}
        resumed = self._panel_result(
            prompt_text=prompt.text,
            text_value="Новый полный ответ после следующего run attempt.",
            limited=False,
            prompt_tokens=79,
            completion_tokens=311,
        )
        async with SessionLocal() as session:
            answer = ModelAnswer(
                run_id=run_id,
                prompt_id=prompt.id,
                provider_key="openai",
                model="test/model",
                mode="web",
                status="failed",
                response_text=historical_raw,
                citations_json=historical_citations,
                usage_json=historical_usage,
                error_message="historical failure",
            )
            session.add(answer)
            await session.flush()
            answer_id = answer.id
            session.add(
                AnswerAnnotation(
                    answer_id=answer_id,
                    annotation_json={"stale": True},
                )
            )
            await session.commit()

        call_started = asyncio.Event()

        async def wait_in_atomic_call(**_kwargs: object) -> SimpleNamespace:
            call_started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        task: asyncio.Task[None] | None = None
        try:
            with (
                patch(
                    "app.services.analyzer.panel_models",
                    return_value=(panel,),
                ),
                patch(
                    "app.services.analyzer.chat",
                    new_callable=AsyncMock,
                    side_effect=wait_in_atomic_call,
                ) as first_worker_chat,
                patch(
                    "app.services.analyzer.update_progress",
                    new_callable=AsyncMock,
                ),
            ):
                task = asyncio.create_task(
                    _run_panel(
                        run_id,
                        [prompt],
                        mode="web",
                        start_percent=40,
                        end_percent=60,
                    )
                )
                await asyncio.wait_for(call_started.wait(), timeout=5)
                first_worker_chat.assert_awaited_once()
                task.cancel()
                outcomes = await asyncio.gather(task, return_exceptions=True)
                self.assertIsInstance(outcomes[0], asyncio.CancelledError)

            async with SessionLocal() as session:
                answer = (
                    await session.execute(
                        select(ModelAnswer).where(ModelAnswer.id == answer_id)
                    )
                ).scalar_one()
                artifacts_before_resume = list(
                    (
                        await session.execute(
                            select(RunArtifact)
                            .where(
                                RunArtifact.run_id == run_id,
                                RunArtifact.prompt_version
                                == PANEL_ATTEMPT_AUDIT_VERSION,
                            )
                            .order_by(RunArtifact.id)
                        )
                    )
                    .scalars()
                    .all()
                )
                annotation_count = (
                    await session.execute(
                        select(func.count(AnswerAnnotation.id)).where(
                            AnswerAnnotation.answer_id == answer_id
                        )
                    )
                ).scalar_one()
            self.assertRegex(
                answer.status,
                r"^running:[0-9a-f]{8}:[0-9a-f]{12}$",
            )
            self.assertEqual(answer.response_text, historical_raw)
            self.assertEqual(annotation_count, 0)
            self.assertEqual(len(artifacts_before_resume), 2)
            imported = [
                row
                for row in artifacts_before_resume
                if row.input_json.get("imported_existing_evidence")
            ]
            self.assertEqual(len(imported), 1)
            self.assertEqual(imported[0].raw_text, historical_raw)
            self.assertEqual(
                imported[0].output_json["annotation"],
                {"stale": True},
            )
            attempted = [
                row
                for row in artifacts_before_resume
                if not row.input_json.get("imported_existing_evidence")
            ]
            self.assertEqual(
                [row.status for row in attempted],
                ["running"],
            )
            self.assertIsNone(attempted[0].raw_text)

            async with SessionLocal() as session:
                await session.execute(
                    update(Run)
                    .where(Run.id == run_id)
                    .values(attempt_count=Run.attempt_count + 1)
                )
                await session.commit()
            with (
                patch(
                    "app.services.analyzer.panel_models",
                    return_value=(panel,),
                ),
                patch(
                    "app.services.analyzer.chat",
                    new_callable=AsyncMock,
                    return_value=resumed,
                ) as resumed_chat,
                patch(
                    "app.services.analyzer.update_progress",
                    new_callable=AsyncMock,
                ),
            ):
                await _run_panel(
                    run_id,
                    [prompt],
                    mode="web",
                    start_percent=40,
                    end_percent=60,
                )
            resumed_chat.assert_awaited_once()
            async with SessionLocal() as session:
                answer = (
                    await session.execute(
                        select(ModelAnswer).where(ModelAnswer.id == answer_id)
                    )
                ).scalar_one()
                all_artifacts = list(
                    (
                        await session.execute(
                            select(RunArtifact)
                            .where(
                                RunArtifact.run_id == run_id,
                                RunArtifact.prompt_version
                                == PANEL_ATTEMPT_AUDIT_VERSION,
                            )
                            .order_by(RunArtifact.id)
                        )
                    )
                    .scalars()
                    .all()
                )
            self.assertEqual(answer.status, "completed")
            self.assertEqual(answer.response_text, resumed.text)
            self.assertEqual(len(all_artifacts), 3)
            self.assertEqual(imported[0].raw_text, historical_raw)
            self.assertEqual(
                [row.status for row in all_artifacts],
                ["completed", "running", "completed"],
            )
        finally:
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            async with SessionLocal() as session:
                await session.execute(delete(Run).where(Run.id == run_id))
                await session.commit()

    async def test_panel_policy_failure_persists_one_atomic_attempt(self) -> None:
        run_id, prompt, panel = await self._single_panel_fixture()
        policy_result = self._panel_result(
            prompt_text=prompt.text,
            text_value="Ответ с нарушенной web-policy.",
            limited=False,
            prompt_tokens=73,
            completion_tokens=931,
        )
        policy_result.usage["_aiv_web_attestation"].update(
            {
                "state": "failed",
                "metric_eligible": False,
                "violations": ["web_search_requests_not_confirmed"],
            }
        )
        failure = OpenRouterPolicyError(
            "policy failure",
            result=policy_result,
        )
        try:
            with (
                patch(
                    "app.services.analyzer.panel_models",
                    return_value=(panel,),
                ),
                patch(
                    "app.services.analyzer.chat",
                    new_callable=AsyncMock,
                    side_effect=failure,
                ) as panel_chat,
                patch(
                    "app.services.analyzer.update_progress",
                    new_callable=AsyncMock,
                ),
                self.assertRaisesRegex(
                    OpenRouterError,
                    "Too few successful web panel responses",
                ),
            ):
                await _run_panel(
                    run_id,
                    [prompt],
                    mode="web",
                    start_percent=40,
                    end_percent=60,
                )

            panel_chat.assert_awaited_once()
            async with SessionLocal() as session:
                answer = (
                    await session.execute(
                        select(ModelAnswer).where(ModelAnswer.run_id == run_id)
                    )
                ).scalar_one()
            self.assertEqual(answer.status, "failed")
            self.assertEqual(answer.response_text, policy_result.text)
            self.assertEqual(answer.usage_json["total_tokens"], 1_004)
            self.assertNotIn("_aiv_panel_retry", answer.usage_json)
            self.assertEqual(
                answer.usage_json["_aiv_panel_contract"]["output_policy"],
                PANEL_OUTPUT_POLICY,
            )
            async with SessionLocal() as session:
                audit = (
                    await session.execute(
                        select(RunArtifact).where(
                            RunArtifact.run_id == run_id,
                            RunArtifact.prompt_version == PANEL_ATTEMPT_AUDIT_VERSION,
                        )
                    )
                ).scalar_one()
            self.assertEqual(audit.status, "failed")
            self.assertEqual(audit.raw_text, policy_result.text)
            self.assertEqual(
                audit.output_json["error_type"],
                "OpenRouterPolicyError",
            )
            metric_rows = await _metric_rows(
                run_id,
                annotation_input_sha256="unused",
            )
            self.assertFalse(metric_rows[0]["metric_eligible"])
        finally:
            async with SessionLocal() as session:
                await session.execute(delete(Run).where(Run.id == run_id))
                await session.commit()

    async def test_panel_paid_response_is_promoted_after_crash_without_post(
        self,
    ) -> None:
        run_id, prompt, panel = await self._single_panel_fixture()
        raw_text = "Содержательный ответ, сохранённый до аварии воркера."
        response_body = {
            "id": "generation-test",
            "model": "test/model",
            "provider": "Test Provider",
            "choices": [
                {
                    "finish_reason": "stop",
                    "native_finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": raw_text,
                        "annotations": [
                            {
                                "type": "url_citation",
                                "url_citation": {
                                    "url": "https://source.example",
                                    "title": "Source",
                                    "content": "Fact",
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 31,
                "completion_tokens": 47,
                "total_tokens": 78,
                "server_tool_use": {"web_search_requests": 1},
            },
            "openrouter_metadata": {},
        }

        class Response:
            status_code = 200

            def json(self) -> dict[str, Any]:
                return copy.deepcopy(response_body)

        class Client:
            posts = 0

            async def __aenter__(self) -> "Client":
                return self

            async def __aexit__(self, *_args: object) -> None:
                return None

            async def post(
                self,
                _url: str,
                *,
                headers: dict[str, str],
                content: bytes,
            ) -> Response:
                self.posts += 1
                self.assert_request = json.loads(content.decode("utf-8"))
                return Response()

        client = Client()
        real_chat = openrouter_service.chat

        async def paid_then_crash(**kwargs: Any) -> Any:
            await real_chat(**kwargs)
            raise asyncio.CancelledError()

        try:
            with (
                patch(
                    "app.services.analyzer.panel_models",
                    return_value=(panel,),
                ),
                patch(
                    "app.services.analyzer.chat",
                    side_effect=paid_then_crash,
                ),
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
                    new_callable=AsyncMock,
                    return_value={
                        "version": OUTPUT_ENVELOPE_VERSION,
                        "policy": OutputTokenPolicy.MODEL_MAX.value,
                        "requested_model": "test/model",
                        "resolution": "test_fixture",
                        "context_length": 128_000,
                        "max_completion_tokens": 32_000,
                    },
                ),
                patch(
                    "app.services.analyzer.update_progress",
                    new_callable=AsyncMock,
                ),
                self.assertRaises(asyncio.CancelledError),
            ):
                await _run_panel(
                    run_id,
                    [prompt],
                    mode="web",
                    start_percent=40,
                    end_percent=60,
                )
            self.assertEqual(client.posts, 1)

            async with SessionLocal() as session:
                await session.execute(
                    update(Run)
                    .where(Run.id == run_id)
                    .values(attempt_count=Run.attempt_count + 1)
                )
                await session.commit()

            with (
                patch(
                    "app.services.analyzer.panel_models",
                    return_value=(panel,),
                ),
                patch(
                    "app.services.analyzer.chat",
                    new_callable=AsyncMock,
                    side_effect=AssertionError("replay must not issue a POST"),
                ) as forbidden_post,
                patch(
                    "app.services.analyzer.update_progress",
                    new_callable=AsyncMock,
                ),
            ):
                await _run_panel(
                    run_id,
                    [prompt],
                    mode="web",
                    start_percent=40,
                    end_percent=60,
                )
            forbidden_post.assert_not_awaited()
            async with SessionLocal() as session:
                answer = (
                    await session.execute(
                        select(ModelAnswer).where(ModelAnswer.run_id == run_id)
                    )
                ).scalar_one()
                artifacts = list(
                    (
                        await session.execute(
                            select(RunArtifact).where(
                                RunArtifact.run_id == run_id,
                                RunArtifact.prompt_version
                                == PANEL_ATTEMPT_AUDIT_VERSION,
                            )
                        )
                    )
                    .scalars()
                    .all()
                )
            self.assertEqual(answer.status, "completed")
            self.assertEqual(answer.response_text, raw_text)
            self.assertEqual(len(artifacts), 1)
            self.assertEqual(artifacts[0].status, "completed")
            self.assertEqual(
                artifacts[0].output_json["provider_event"]["status"],
                "accepted",
            )
        finally:
            async with SessionLocal() as session:
                await session.execute(delete(Run).where(Run.id == run_id))
                await session.commit()

    async def test_panel_contract_failure_persists_one_atomic_attempt(
        self,
    ) -> None:
        run_id, prompt, panel = await self._single_panel_fixture()
        contract_result = self._panel_result(
            prompt_text=prompt.text,
            text_value="Незавершённый ответ.",
            limited=False,
            output_complete=False,
            prompt_tokens=79,
            completion_tokens=1_207,
        )
        failure = OpenRouterResponseContractError(
            "response contract failure",
            result=contract_result,
        )
        try:
            with (
                patch(
                    "app.services.analyzer.panel_models",
                    return_value=(panel,),
                ),
                patch(
                    "app.services.analyzer.chat",
                    new_callable=AsyncMock,
                    side_effect=failure,
                ) as panel_chat,
                patch(
                    "app.services.analyzer.update_progress",
                    new_callable=AsyncMock,
                ),
                self.assertRaisesRegex(
                    OpenRouterError,
                    "Too few successful web panel responses",
                ),
            ):
                await _run_panel(
                    run_id,
                    [prompt],
                    mode="web",
                    start_percent=40,
                    end_percent=60,
                )

            panel_chat.assert_awaited_once()
            async with SessionLocal() as session:
                answer = (
                    await session.execute(
                        select(ModelAnswer).where(ModelAnswer.run_id == run_id)
                    )
                ).scalar_one()
            self.assertEqual(answer.status, "failed")
            self.assertEqual(answer.response_text, contract_result.text)
            self.assertEqual(answer.usage_json["total_tokens"], 1_286)
            self.assertNotIn("_aiv_panel_retry", answer.usage_json)
            async with SessionLocal() as session:
                audit = (
                    await session.execute(
                        select(RunArtifact).where(
                            RunArtifact.run_id == run_id,
                            RunArtifact.prompt_version == PANEL_ATTEMPT_AUDIT_VERSION,
                        )
                    )
                ).scalar_one()
            self.assertEqual(audit.status, "failed")
            self.assertEqual(audit.raw_text, contract_result.text)
            self.assertFalse(audit.output_json["transport"]["output_limited"])
            self.assertFalse(audit.output_json["transport"]["output_complete"])
            self.assertEqual(
                audit.output_json["error_type"],
                "OpenRouterResponseContractError",
            )
            metric_rows = await _metric_rows(
                run_id,
                annotation_input_sha256="unused",
            )
            self.assertFalse(metric_rows[0]["metric_eligible"])
        finally:
            async with SessionLocal() as session:
                await session.execute(delete(Run).where(Run.id == run_id))
                await session.commit()

    async def test_panel_transport_failure_persists_empty_atomic_audit(
        self,
    ) -> None:
        run_id, prompt, panel = await self._single_panel_fixture()
        failure = OpenRouterError("atomic transport failure")
        try:
            with (
                patch(
                    "app.services.analyzer.panel_models",
                    return_value=(panel,),
                ),
                patch(
                    "app.services.analyzer.chat",
                    new_callable=AsyncMock,
                    side_effect=failure,
                ) as panel_chat,
                patch(
                    "app.services.analyzer.update_progress",
                    new_callable=AsyncMock,
                ),
                self.assertRaisesRegex(
                    OpenRouterError,
                    "Too few successful web panel responses",
                ),
            ):
                await _run_panel(
                    run_id,
                    [prompt],
                    mode="web",
                    start_percent=40,
                    end_percent=60,
                )

            panel_chat.assert_awaited_once()
            async with SessionLocal() as session:
                answer = (
                    await session.execute(
                        select(ModelAnswer).where(ModelAnswer.run_id == run_id)
                    )
                ).scalar_one()
            self.assertEqual(answer.status, "failed")
            self.assertIsNone(answer.response_text)
            self.assertIsNone(answer.usage_json)
            self.assertIn("atomic transport failure", answer.error_message)
            async with SessionLocal() as session:
                audit = (
                    await session.execute(
                        select(RunArtifact).where(
                            RunArtifact.run_id == run_id,
                            RunArtifact.prompt_version == PANEL_ATTEMPT_AUDIT_VERSION,
                        )
                    )
                ).scalar_one()
            self.assertEqual(audit.status, "failed")
            self.assertIsNone(audit.raw_text)
            self.assertEqual(audit.input_json["max_tokens"], None)
            self.assertEqual(
                audit.input_json["output_policy"],
                PANEL_OUTPUT_POLICY,
            )
            self.assertEqual(
                audit.output_json["error_type"],
                "OpenRouterError",
            )
            self.assertIsNone(audit.output_json["response_text_sha256"])
            self.assertEqual(audit.output_json["response_char_count"], 0)
            metric_rows = await _metric_rows(
                run_id,
                annotation_input_sha256="unused",
            )
            self.assertFalse(metric_rows[0]["metric_eligible"])
        finally:
            async with SessionLocal() as session:
                await session.execute(delete(Run).where(Run.id == run_id))
                await session.commit()

    async def test_concurrent_panel_workers_issue_one_paid_call_per_cell(
        self,
    ) -> None:
        run_id, prompt, panel = await self._single_panel_fixture()
        result = self._panel_result(
            prompt_text=prompt.text,
            text_value="Единственный завершённый ответ.",
            limited=False,
            prompt_tokens=71,
            completion_tokens=113,
        )
        started = asyncio.Event()
        release = asyncio.Event()
        tasks: list[asyncio.Task[None]] = []

        async def one_success(**_kwargs: object) -> SimpleNamespace:
            started.set()
            await release.wait()
            return result

        try:
            with (
                patch(
                    "app.services.analyzer.panel_models",
                    return_value=(panel,),
                ),
                patch(
                    "app.services.analyzer.chat",
                    new_callable=AsyncMock,
                    side_effect=one_success,
                ) as panel_chat,
                patch(
                    "app.services.analyzer.update_progress",
                    new_callable=AsyncMock,
                ),
            ):
                tasks = [
                    asyncio.create_task(
                        _run_panel(
                            run_id,
                            [prompt],
                            mode="web",
                            start_percent=40,
                            end_percent=60,
                        )
                    )
                    for _ in range(2)
                ]
                await asyncio.wait_for(started.wait(), timeout=5)
                await asyncio.sleep(0.05)
                self.assertEqual(panel_chat.await_count, 1)
                release.set()
                await asyncio.gather(*tasks)

            self.assertEqual(panel_chat.await_count, 1)
            async with SessionLocal() as session:
                answer = (
                    await session.execute(
                        select(ModelAnswer).where(ModelAnswer.run_id == run_id)
                    )
                ).scalar_one()
            self.assertEqual(answer.status, "completed")
            self.assertEqual(answer.response_text, result.text)
        finally:
            release.set()
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            async with SessionLocal() as session:
                await session.execute(delete(Run).where(Run.id == run_id))
                await session.commit()

    async def test_later_run_attempt_reclaims_cell_and_rejects_stale_token(
        self,
    ) -> None:
        run_id, prompt, panel = await self._single_panel_fixture()
        stale = self._panel_result(
            prompt_text=prompt.text,
            text_value="Устаревший worker не должен сохранить этот ответ.",
            limited=False,
            prompt_tokens=61,
            completion_tokens=149,
        )
        resumed = self._panel_result(
            prompt_text=prompt.text,
            text_value="Ответ нового run attempt.",
            limited=False,
            prompt_tokens=67,
            completion_tokens=151,
        )
        stale_started = asyncio.Event()
        release_stale = asyncio.Event()
        call_count = 0
        tasks: list[asyncio.Task[None]] = []

        async def stale_then_resumed(**_kwargs: object) -> SimpleNamespace:
            nonlocal call_count
            call_index = call_count
            call_count += 1
            if call_index == 0:
                stale_started.set()
                await release_stale.wait()
                return stale
            return resumed

        try:
            with (
                patch(
                    "app.services.analyzer.panel_models",
                    return_value=(panel,),
                ),
                patch(
                    "app.services.analyzer.chat",
                    new_callable=AsyncMock,
                    side_effect=stale_then_resumed,
                ) as panel_chat,
                patch(
                    "app.services.analyzer.update_progress",
                    new_callable=AsyncMock,
                ),
            ):
                tasks.append(
                    asyncio.create_task(
                        _run_panel(
                            run_id,
                            [prompt],
                            mode="web",
                            start_percent=40,
                            end_percent=60,
                        )
                    )
                )
                await asyncio.wait_for(stale_started.wait(), timeout=5)
                async with SessionLocal() as session:
                    await session.execute(
                        update(Run)
                        .where(Run.id == run_id)
                        .values(attempt_count=Run.attempt_count + 1)
                    )
                    await session.commit()
                tasks.append(
                    asyncio.create_task(
                        _run_panel(
                            run_id,
                            [prompt],
                            mode="web",
                            start_percent=40,
                            end_percent=60,
                        )
                    )
                )
                await asyncio.wait_for(tasks[1], timeout=5)
                release_stale.set()
                await asyncio.wait_for(tasks[0], timeout=5)

            self.assertEqual(panel_chat.await_count, 2)
            async with SessionLocal() as session:
                answer = (
                    await session.execute(
                        select(ModelAnswer).where(ModelAnswer.run_id == run_id)
                    )
                ).scalar_one()
            self.assertEqual(answer.status, "completed")
            self.assertEqual(answer.response_text, resumed.text)
            self.assertIsNone(answer.error_message)
        finally:
            release_stale.set()
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
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
                response_text="Сохранённый ответ",
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
            self.assertTrue(provenance["web_attestation"]["metric_eligible"])

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

    async def test_panel_resume_checkpoint_accepts_supported_prompt_versions(
        self,
    ) -> None:
        versions = (PROMPT_SET_VERSION, sorted(LEGACY_PROMPT_SET_VERSIONS)[-1])
        for prompt_version in versions:
            with self.subTest(prompt_version=prompt_version):
                (
                    run_id,
                    _items,
                    prompt_ids,
                    _answer_id,
                ) = await self._create_panel_checkpoint(
                    prompt_version=prompt_version,
                    answer_status="pending",
                )
                try:
                    checkpoint = await _load_panel_resume_checkpoint(run_id)
                    self.assertIsNotNone(checkpoint)
                    self.assertEqual(
                        [prompt.id for prompt in checkpoint or []],
                        prompt_ids,
                    )
                finally:
                    await self._delete_run(run_id)

    async def test_panel_resume_checkpoint_validates_running_claim_token(
        self,
    ) -> None:
        run_id, _items, _prompt_ids, answer_id = await self._create_panel_checkpoint(
            answer_status="running:00000001:0123456789ab"
        )
        try:
            checkpoint = await _load_panel_resume_checkpoint(run_id)
            self.assertIsNotNone(checkpoint)
            async with SessionLocal() as session:
                await session.execute(
                    update(ModelAnswer)
                    .where(ModelAnswer.id == answer_id)
                    .values(status="running:00000001:not-a-valid-token")
                )
                await session.commit()
            with self.assertRaisesRegex(
                PanelCheckpointMismatchError,
                "answer_status_invalid",
            ):
                await _load_panel_resume_checkpoint(run_id)
        finally:
            await self._delete_run(run_id)

    async def test_full_panel_corpus_receipt_seals_all_nine_prompts_and_cells(
        self,
    ) -> None:
        run_id, answer_ids = await self._create_full_panel_corpus()
        try:
            receipt = await _seal_or_validate_panel_corpus_receipt(
                run_id,
                allow_legacy_baseline=False,
            )
            self.assertEqual(receipt["version"], PANEL_CORPUS_RECEIPT_VERSION)
            self.assertEqual(receipt["proof_scope"], "normal_panel_completion")
            self.assertTrue(receipt["historical_integrity_proven"])
            self.assertEqual(receipt["prompt_count"], 9)
            self.assertEqual(receipt["expected_cell_count"], 81)
            self.assertEqual(receipt["actual_cell_count"], 81)
            self.assertEqual(len(receipt["prompts"]), 9)
            self.assertEqual(len(receipt["cells"]), 81)
            self.assertEqual(
                {item["role"] for item in receipt["prompts"]},
                {"unbranded_discovery", "brand_diagnostic"},
            )
            first = receipt["cells"][0]
            self.assertGreater(first["response_utf8_bytes"], 0)
            self.assertEqual(len(first["response_sha256"]), 64)
            self.assertEqual(len(first["citations_sha256"]), 64)
            self.assertEqual(len(first["usage_sha256"]), 64)

            # A second call validates the existing seal; it cannot reseal a
            # changed corpus under a fresh digest.
            self.assertEqual(
                await _seal_or_validate_panel_corpus_receipt(
                    run_id,
                    allow_legacy_baseline=False,
                ),
                receipt,
            )
            async with SessionLocal() as session:
                await session.execute(
                    update(ModelAnswer)
                    .where(ModelAnswer.id == answer_ids[0])
                    .values(response_text="Подменённый raw-ответ")
                )
                await session.commit()
            with self.assertRaisesRegex(
                PanelCheckpointMismatchError,
                "persisted_corpus_changed",
            ):
                await _validate_panel_corpus_receipt_if_present(run_id)
        finally:
            await self._delete_run(run_id)

    async def test_sealed_historical_topology_survives_current_memory_lane_swap(
        self,
    ) -> None:
        run_id, _answer_ids = await self._create_full_panel_corpus()
        original_panels = panel_models()
        drifted_panels = tuple(
            PanelModel(
                key=panel.key,
                label=panel.label,
                model=panel.model,
                memory_model=(
                    panel.model
                    if panel.key == "perplexity"
                    else None
                    if panel.key == "claude"
                    else panel.memory_model
                ),
            )
            for panel in original_panels
        )
        try:
            receipt = await _seal_or_validate_panel_corpus_receipt(
                run_id,
                allow_legacy_baseline=False,
            )
            self.assertEqual(receipt["actual_cell_count"], 81)

            with patch(
                "app.services.analyzer.panel_models",
                return_value=drifted_panels,
            ):
                checkpoint = await _load_panel_resume_checkpoint(run_id)
                expected_cells = await _expected_corpus_cells(run_id, [])
                self.assertTrue(
                    await _validate_panel_corpus_receipt_if_present(run_id)
                )

            self.assertEqual(len(checkpoint or []), 9)
            memory_providers = {
                str(cell["provider_key"])
                for cell in expected_cells
                if cell["mode"] == "memory"
            }
            self.assertIn("claude", memory_providers)
            self.assertNotIn("perplexity", memory_providers)
            self.assertEqual(len(expected_cells), 81)
        finally:
            await self._delete_run(run_id)

    async def test_current_lane_cannot_enter_a_sealed_historical_grid(
        self,
    ) -> None:
        run_id, _answer_ids = await self._create_full_panel_corpus()
        original_panels = panel_models()
        drifted_panels = tuple(
            PanelModel(
                key=panel.key,
                label=panel.label,
                model=panel.model,
                memory_model=(
                    panel.model
                    if panel.key == "perplexity"
                    else None
                    if panel.key == "claude"
                    else panel.memory_model
                ),
            )
            for panel in original_panels
        )
        try:
            await _seal_or_validate_panel_corpus_receipt(
                run_id,
                allow_legacy_baseline=False,
            )
            async with SessionLocal() as session:
                prompt_id = (
                    await session.execute(
                        select(VisibilityPrompt.id)
                        .where(VisibilityPrompt.run_id == run_id)
                        .order_by(VisibilityPrompt.sequence)
                        .limit(1)
                    )
                ).scalar_one()
                perplexity = next(
                    panel for panel in drifted_panels if panel.key == "perplexity"
                )
                session.add(
                    ModelAnswer(
                        run_id=run_id,
                        prompt_id=prompt_id,
                        provider_key="perplexity",
                        model=str(perplexity.memory_model),
                        mode="memory",
                        status="completed",
                        response_text="Неожиданная ячейка новой конфигурации.",
                    )
                )
                await session.commit()

            with (
                patch(
                    "app.services.analyzer.panel_models",
                    return_value=drifted_panels,
                ),
                self.assertRaisesRegex(
                    PanelCheckpointMismatchError,
                    "sealed_answer_grid_changed",
                ),
            ):
                await _load_panel_resume_checkpoint(run_id)
        finally:
            await self._delete_run(run_id)

    async def test_legacy_panel_corpus_baseline_never_claims_historical_proof(
        self,
    ) -> None:
        run_id, _answer_ids = await self._create_full_panel_corpus()
        try:
            receipt = await _seal_or_validate_panel_corpus_receipt(
                run_id,
                allow_legacy_baseline=True,
            )
            self.assertEqual(
                receipt["proof_scope"],
                "legacy_reprocess_baseline",
            )
            self.assertFalse(receipt["historical_integrity_proven"])
            async with SessionLocal() as session:
                artifacts = {
                    artifact.artifact_key: artifact
                    for artifact in (
                        (
                            await session.execute(
                                select(RunArtifact).where(
                                    RunArtifact.run_id == run_id,
                                    RunArtifact.artifact_key.in_(
                                        (
                                            PANEL_CORPUS_RECEIPT_KEY,
                                            "legacy_panel_corpus_baseline_audit",
                                        )
                                    ),
                                )
                            )
                        )
                        .scalars()
                        .all()
                    )
                }
            self.assertEqual(
                set(artifacts),
                {
                    PANEL_CORPUS_RECEIPT_KEY,
                    "legacy_panel_corpus_baseline_audit",
                },
            )
            self.assertFalse(
                artifacts["legacy_panel_corpus_baseline_audit"].output_json[
                    "historical_integrity_proven"
                ]
            )
            self.assertEqual(
                artifacts["legacy_panel_corpus_baseline_audit"].output_json[
                    "panel_corpus_receipt_sha256"
                ],
                receipt["receipt_sha256"],
            )
            async with SessionLocal() as session:
                await session.execute(
                    update(RunArtifact)
                    .where(
                        RunArtifact.run_id == run_id,
                        RunArtifact.artifact_key
                        == "legacy_panel_corpus_baseline_audit",
                    )
                    .values(output_json={"historical_integrity_proven": True})
                )
                await session.commit()
            with self.assertRaisesRegex(
                PanelCheckpointMismatchError,
                "legacy_baseline_audit",
            ):
                await _seal_or_validate_panel_corpus_receipt(
                    run_id,
                    allow_legacy_baseline=True,
                )
        finally:
            await self._delete_run(run_id)

    async def test_panel_corpus_receipt_detects_citation_and_usage_tampering(
        self,
    ) -> None:
        run_id, answer_ids = await self._create_full_panel_corpus()
        try:
            await _seal_or_validate_panel_corpus_receipt(
                run_id,
                allow_legacy_baseline=False,
            )
            async with SessionLocal() as session:
                answer = await session.get(ModelAnswer, answer_ids[0])
                assert answer is not None
                original_citations = copy.deepcopy(answer.citations_json)
                original_usage = copy.deepcopy(answer.usage_json)
                answer.citations_json = [{"url": "https://tampered.example/"}]
                await session.commit()
            with self.assertRaisesRegex(
                PanelCheckpointMismatchError,
                "persisted_corpus_changed",
            ):
                await _validate_panel_corpus_receipt_if_present(run_id)

            async with SessionLocal() as session:
                answer = await session.get(ModelAnswer, answer_ids[0])
                assert answer is not None
                answer.citations_json = original_citations
                answer.usage_json = {"total_tokens": 999_999}
                await session.commit()
            with self.assertRaisesRegex(
                PanelCheckpointMismatchError,
                "persisted_corpus_changed",
            ):
                await _validate_panel_corpus_receipt_if_present(run_id)

            async with SessionLocal() as session:
                answer = await session.get(ModelAnswer, answer_ids[0])
                assert answer is not None
                answer.usage_json = original_usage
                await session.commit()
            self.assertTrue(await _validate_panel_corpus_receipt_if_present(run_id))
        finally:
            await self._delete_run(run_id)

    async def test_analyze_run_never_reopens_a_sealed_panel_corpus(self) -> None:
        run_id, _answer_ids = await self._create_full_panel_corpus()
        await _seal_or_validate_panel_corpus_receipt(
            run_id,
            allow_legacy_baseline=False,
        )
        panel = AsyncMock()
        finish = AsyncMock()
        failure = AsyncMock()
        try:
            with (
                patch(
                    "app.services.analyzer.update_progress",
                    new=AsyncMock(),
                ),
                patch(
                    "app.services.analyzer.apply_ua_conditional_block",
                    new=AsyncMock(),
                ),
                patch(
                    "app.services.analyzer._prepare_analysis_foundation",
                    new=AsyncMock(
                        return_value=(
                            {"score": 100},
                            {"summary": "ok"},
                            {"brand_name": "Example"},
                            {"requested_site": {"domain": "example.com"}},
                        )
                    ),
                ),
                patch(
                    "app.services.analyzer._validate_panel_foundation_resume",
                    new=AsyncMock(),
                ),
                patch("app.services.analyzer._run_panel", new=panel),
                patch(
                    "app.services.analyzer._save_answer_set_receipt",
                    new=AsyncMock(),
                ),
                patch(
                    "app.services.analyzer._finish_saved_answer_analysis",
                    new=finish,
                ),
                patch("app.services.analyzer.fail_run", new=failure),
            ):
                await analyze_run(run_id)

            panel.assert_not_awaited()
            finish.assert_awaited_once()
            failure.assert_not_awaited()
        finally:
            await self._delete_run(run_id)

    async def test_panel_checkpoint_rejects_unknown_provider_lane(self) -> None:
        run_id, _items, _prompt_ids, answer_id = await self._create_panel_checkpoint(
            answer_status="pending"
        )
        try:
            async with SessionLocal() as session:
                await session.execute(
                    update(ModelAnswer)
                    .where(ModelAnswer.id == answer_id)
                    .values(provider_key="unknown-provider")
                )
                await session.commit()
            with self.assertRaisesRegex(
                PanelCheckpointMismatchError,
                "unknown_provider_mode",
            ):
                await _load_panel_resume_checkpoint(run_id)
        finally:
            await self._delete_run(run_id)

    async def test_panel_checkpoint_rejects_mixed_models_inside_one_lane(
        self,
    ) -> None:
        run_id, _items, prompt_ids, _answer_id = await self._create_panel_checkpoint(
            answer_status="pending"
        )
        try:
            async with SessionLocal() as session:
                session.add(
                    ModelAnswer(
                        run_id=run_id,
                        prompt_id=prompt_ids[1],
                        provider_key="openai",
                        model="another/model",
                        mode="web",
                        status="pending",
                    )
                )
                await session.commit()
            with self.assertRaisesRegex(
                PanelCheckpointMismatchError,
                "model_lane_mismatch",
            ):
                await _load_panel_resume_checkpoint(run_id)
        finally:
            await self._delete_run(run_id)

    async def test_panel_checkpoint_resumes_75_completed_and_6_failed_rows(
        self,
    ) -> None:
        run_id = f"test-panel-81-{uuid.uuid4()}"
        items = self._checkpoint_prompt_items()
        panels = tuple(
            PanelModel(
                key=f"provider-{index}",
                label=f"Provider {index}",
                model=f"test/web-{index}",
                memory_model=(f"test/memory-{index}" if index < 4 else None),
            )
            for index in range(5)
        )
        completed_snapshot: dict[int, str] = {}
        failed_ids: list[int] = []
        async with SessionLocal() as session:
            session.add(
                Run(
                    id=run_id,
                    domain="example.com",
                    status=RunStatus.analyzing,
                    config_json={},
                )
            )
            prompts: list[VisibilityPrompt] = []
            for sequence, item in enumerate(items, start=1):
                prompt = VisibilityPrompt(
                    run_id=run_id,
                    prompt_key=str(item["prompt_key"]),
                    intent_class=str(item["intent_class"]),
                    role=str(item["role"]),
                    text=str(item["text"]),
                    rationale=str(item["rationale"]),
                    sequence=sequence,
                )
                session.add(prompt)
                prompts.append(prompt)
            await session.flush()
            session.add(
                RunArtifact(
                    run_id=run_id,
                    stage_key="scenario_design",
                    artifact_key="prompt_set",
                    status="completed",
                    prompt_version=PROMPT_SET_VERSION,
                    output_json={"prompts": copy.deepcopy(items)},
                )
            )
            cell_index = 0
            for mode in ("web", "memory"):
                for prompt in prompts:
                    for panel in panels:
                        model = panel.model if mode == "web" else panel.memory_model
                        if model is None:
                            continue
                        cell_index += 1
                        is_failed = cell_index <= 6
                        raw = (
                            f"Старый failed raw {cell_index}"
                            if is_failed
                            else f"Неизменяемый completed raw {cell_index}"
                        )
                        answer = ModelAnswer(
                            run_id=run_id,
                            prompt_id=prompt.id,
                            provider_key=panel.key,
                            model=model,
                            mode=mode,
                            status="failed" if is_failed else "completed",
                            response_text=raw,
                            citations_json=[{"url": f"https://e/{cell_index}"}],
                            usage_json={"total_tokens": cell_index},
                            error_message=("old failure" if is_failed else None),
                        )
                        session.add(answer)
                        await session.flush()
                        if is_failed:
                            failed_ids.append(answer.id)
                        else:
                            completed_snapshot[answer.id] = json.dumps(
                                {
                                    "response_text": answer.response_text,
                                    "citations_json": answer.citations_json,
                                    "usage_json": answer.usage_json,
                                    "model": answer.model,
                                },
                                sort_keys=True,
                                ensure_ascii=False,
                            )
            await session.commit()
        self.assertEqual(cell_index, 81)
        self.assertEqual(len(completed_snapshot), 75)
        self.assertEqual(len(failed_ids), 6)

        recovered_calls = 0

        async def recovered_result(**_kwargs: object) -> SimpleNamespace:
            nonlocal recovered_calls
            recovered_calls += 1
            return SimpleNamespace(
                text=f"Восстановленный ответ {recovered_calls}",
                citations=[],
                usage={"total_tokens": 100 + recovered_calls},
                request_policy={"sha256": "policy", "policy": "required"},
                web_attestation={"metric_eligible": True},
            )

        try:
            with (
                patch(
                    "app.services.analyzer.panel_models",
                    return_value=panels,
                ),
                patch(
                    "app.services.analyzer.chat",
                    new_callable=AsyncMock,
                    side_effect=recovered_result,
                ) as panel_chat,
                patch(
                    "app.services.analyzer.update_progress",
                    new_callable=AsyncMock,
                ),
            ):
                checkpoint = await _load_panel_resume_checkpoint(run_id)
                self.assertEqual(len(checkpoint or []), 9)
                await _run_panel(
                    run_id,
                    checkpoint or [],
                    mode="web",
                    start_percent=38,
                    end_percent=64,
                )
                await _run_panel(
                    run_id,
                    checkpoint or [],
                    mode="memory",
                    start_percent=65,
                    end_percent=72,
                )
            self.assertEqual(panel_chat.await_count, 6)
            async with SessionLocal() as session:
                answers = list(
                    (
                        await session.execute(
                            select(ModelAnswer).where(ModelAnswer.run_id == run_id)
                        )
                    )
                    .scalars()
                    .all()
                )
                imported_count = (
                    await session.execute(
                        select(func.count(RunArtifact.id)).where(
                            RunArtifact.run_id == run_id,
                            RunArtifact.prompt_version == PANEL_ATTEMPT_AUDIT_VERSION,
                            RunArtifact.artifact_key.contains(
                                "panel_attempt_import_",
                                autoescape=True,
                            ),
                        )
                    )
                ).scalar_one()
            self.assertEqual(len(answers), 81)
            self.assertTrue(all(row.status == "completed" for row in answers))
            self.assertEqual(imported_count, 6)
            for row in answers:
                if row.id not in completed_snapshot:
                    continue
                self.assertEqual(
                    json.dumps(
                        {
                            "response_text": row.response_text,
                            "citations_json": row.citations_json,
                            "usage_json": row.usage_json,
                            "model": row.model,
                        },
                        sort_keys=True,
                        ensure_ascii=False,
                    ),
                    completed_snapshot[row.id],
                )
        finally:
            await self._delete_run(run_id)

    async def test_analyze_run_reuses_exact_panel_checkpoint(self) -> None:
        run_id, _items, prompt_ids, _answer_id = await self._create_panel_checkpoint(
            answer_status="pending"
        )
        market = AsyncMock()
        generate = AsyncMock()
        persist = AsyncMock()
        panel = AsyncMock()
        finish = AsyncMock()
        fail = AsyncMock()
        semantic_review = AsyncMock()
        try:
            with (
                patch(
                    "app.services.analyzer.update_progress",
                    new=AsyncMock(),
                ),
                patch(
                    "app.services.analyzer.apply_ua_conditional_block",
                    new=AsyncMock(),
                ),
                patch(
                    "app.services.analyzer._prepare_analysis_foundation",
                    new=AsyncMock(
                        return_value=(
                            {"score": 100},
                            {"summary": "ok"},
                            {"brand_name": "Example"},
                            {"requested_site": {"domain": "example.com"}},
                        )
                    ),
                ),
                patch("app.services.analyzer._market_research", new=market),
                patch("app.services.analyzer._generate_prompt_set", new=generate),
                patch("app.services.analyzer._persist_prompts", new=persist),
                patch("app.services.analyzer._run_panel", new=panel),
                patch(
                    "app.services.analyzer._seal_or_validate_panel_corpus_receipt",
                    new=AsyncMock(),
                ),
                patch(
                    "app.services.analyzer._save_answer_set_receipt",
                    new=AsyncMock(),
                ),
                patch(
                    "app.services.analyzer._finish_saved_answer_analysis",
                    new=finish,
                ),
                patch("app.services.analyzer.fail_run", new=fail),
                patch(
                    "app.services.analyzer._review_prompt_set_semantics",
                    new=semantic_review,
                ),
            ):
                await analyze_run(run_id)

            market.assert_not_awaited()
            generate.assert_not_awaited()
            persist.assert_not_awaited()
            semantic_review.assert_not_awaited()
            self.assertEqual(panel.await_count, 2)
            self.assertEqual(
                [prompt.id for prompt in panel.await_args_list[0].args[1]],
                prompt_ids,
            )
            self.assertEqual(panel.await_args_list[0].kwargs["mode"], "web")
            self.assertEqual(panel.await_args_list[1].kwargs["mode"], "memory")
            finish.assert_awaited_once()
            fail.assert_not_awaited()
        finally:
            await self._delete_run(run_id)

    async def test_analyze_run_checkpoint_mismatch_fails_before_writes(
        self,
    ) -> None:
        run_id, items, _prompt_ids, answer_id = await self._create_panel_checkpoint(
            artifact_mismatch=True,
            with_annotation=True,
        )
        foundation = AsyncMock()
        market = AsyncMock()
        generate = AsyncMock()
        persist = AsyncMock()
        panel = AsyncMock()
        finish = AsyncMock()
        fail = AsyncMock()
        try:
            with (
                patch(
                    "app.services.analyzer.update_progress",
                    new=AsyncMock(),
                ),
                patch(
                    "app.services.analyzer.apply_ua_conditional_block",
                    new=AsyncMock(),
                ),
                patch(
                    "app.services.analyzer._prepare_analysis_foundation",
                    new=foundation,
                ),
                patch("app.services.analyzer._market_research", new=market),
                patch("app.services.analyzer._generate_prompt_set", new=generate),
                patch("app.services.analyzer._persist_prompts", new=persist),
                patch("app.services.analyzer._run_panel", new=panel),
                patch(
                    "app.services.analyzer._seal_or_validate_panel_corpus_receipt",
                    new=AsyncMock(),
                ),
                patch(
                    "app.services.analyzer._save_answer_set_receipt",
                    new=AsyncMock(),
                ),
                patch(
                    "app.services.analyzer._finish_saved_answer_analysis",
                    new=finish,
                ),
                patch("app.services.analyzer.fail_run", new=fail),
            ):
                await analyze_run(run_id)

            foundation.assert_not_awaited()
            market.assert_not_awaited()
            generate.assert_not_awaited()
            persist.assert_not_awaited()
            panel.assert_not_awaited()
            finish.assert_not_awaited()
            fail.assert_awaited_once()
            self.assertIn("проверку целостности", fail.await_args.args[1])
            self.assertEqual(
                fail.await_args.kwargs,
                {"failure_stage": "integrity_review_required"},
            )

            async with SessionLocal() as session:
                saved_prompt = (
                    (
                        await session.execute(
                            select(VisibilityPrompt)
                            .where(VisibilityPrompt.run_id == run_id)
                            .order_by(VisibilityPrompt.sequence)
                        )
                    )
                    .scalars()
                    .first()
                )
                saved_answer = await session.get(ModelAnswer, answer_id)
                annotation_count = (
                    await session.execute(
                        select(func.count())
                        .select_from(AnswerAnnotation)
                        .where(AnswerAnnotation.answer_id == answer_id)
                    )
                ).scalar_one()
            self.assertEqual(saved_prompt.text, items[0]["text"])
            self.assertEqual(saved_answer.response_text, "Сохранённый raw-ответ")
            self.assertEqual(annotation_count, 1)
        finally:
            await self._delete_run(run_id)

    async def test_analyze_run_without_answers_uses_generation_path(self) -> None:
        run_id = f"test-no-panel-checkpoint-{uuid.uuid4()}"
        async with SessionLocal() as session:
            session.add(
                Run(
                    id=run_id,
                    domain="example.com",
                    status=RunStatus.analyzing,
                    config_json={},
                )
            )
            await session.commit()
        prompt_set = {"prompts": self._checkpoint_prompt_items()}
        market = AsyncMock(return_value={"status": "ready"})
        generate = AsyncMock(return_value=prompt_set)
        persist = AsyncMock(return_value=[])
        panel = AsyncMock()
        finish = AsyncMock()
        fail = AsyncMock()
        save_foundation = AsyncMock()
        try:
            with (
                patch(
                    "app.services.analyzer.update_progress",
                    new=AsyncMock(),
                ),
                patch(
                    "app.services.analyzer.apply_ua_conditional_block",
                    new=AsyncMock(),
                ),
                patch(
                    "app.services.analyzer._prepare_analysis_foundation",
                    new=AsyncMock(
                        return_value=(
                            {"score": 100},
                            {"summary": "ok"},
                            {"brand_name": "Example"},
                            {"requested_site": {"domain": "example.com"}},
                        )
                    ),
                ),
                patch("app.services.analyzer._market_research", new=market),
                patch("app.services.analyzer._generate_prompt_set", new=generate),
                patch("app.services.analyzer._persist_prompts", new=persist),
                patch(
                    "app.services.analyzer._save_prompt_foundation",
                    new=save_foundation,
                ),
                patch("app.services.analyzer._run_panel", new=panel),
                patch(
                    "app.services.analyzer._seal_or_validate_panel_corpus_receipt",
                    new=AsyncMock(),
                ),
                patch(
                    "app.services.analyzer._save_answer_set_receipt",
                    new=AsyncMock(),
                ),
                patch(
                    "app.services.analyzer._finish_saved_answer_analysis",
                    new=finish,
                ),
                patch("app.services.analyzer.fail_run", new=fail),
            ):
                await analyze_run(run_id)

            market.assert_awaited_once()
            generate.assert_awaited_once()
            save_foundation.assert_awaited_once()
            persist.assert_awaited_once_with(run_id, prompt_set)
            self.assertEqual(panel.await_count, 2)
            finish.assert_awaited_once()
            fail.assert_not_awaited()
        finally:
            await self._delete_run(run_id)

    async def test_changed_prompt_preserves_panel_answers_and_annotations(
        self,
    ) -> None:
        run_id, items, prompt_ids, answer_id = await self._create_panel_checkpoint(
            with_annotation=True
        )
        try:
            persisted = await _persist_prompts(
                run_id,
                {"prompts": copy.deepcopy(items)},
            )
            self.assertEqual([prompt.id for prompt in persisted], prompt_ids)

            changed_items = copy.deepcopy(items)
            changed_items[0]["text"] = "Новый пользовательский запрос"
            with self.assertRaises(PanelCheckpointMismatchError):
                await _persist_prompts(run_id, {"prompts": changed_items})

            async with SessionLocal() as session:
                saved_prompt = await session.get(VisibilityPrompt, prompt_ids[0])
                saved_answer = await session.get(ModelAnswer, answer_id)
                annotation_count = (
                    await session.execute(
                        select(func.count())
                        .select_from(AnswerAnnotation)
                        .where(AnswerAnnotation.answer_id == answer_id)
                    )
                ).scalar_one()
            self.assertEqual(saved_prompt.text, items[0]["text"])
            self.assertEqual(saved_answer.response_text, "Сохранённый raw-ответ")
            self.assertEqual(annotation_count, 1)
        finally:
            await self._delete_run(run_id)

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
            session.add(
                AnswerAnnotation(answer_id=answer.id, annotation_json={"valid": True})
            )
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

    async def test_create_run_accepts_only_one_domain_and_uses_fixed_config(
        self,
    ) -> None:
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
                        select(Run.status, Run.resume_count).where(Run.id == run_id)
                    )
                ).one()
            self.assertEqual(status, RunStatus.pending)
            self.assertEqual(resume_count, 1)
        finally:
            async with SessionLocal() as session:
                await session.execute(delete(Run).where(Run.id == run_id))
                await session.commit()

    async def test_completed_run_retry_is_rejected_without_mutating_evidence(
        self,
    ) -> None:
        run_id = f"test-completed-retry-{uuid.uuid4()}"
        async with SessionLocal() as session:
            run = Run(
                id=run_id,
                domain="completed.example.com",
                status=RunStatus.completed,
                config_json={"pipeline_version": "immutable-test"},
                progress_current=100,
                progress_total=100,
                progress_percent=100,
                stage_key="report",
                stage_label="Отчёт готов",
                state_revision=7,
                resume_count=2,
                analysis_markdown="# Готовый отчёт",
                report_json={"headline": "Сохранённый отчёт"},
            )
            session.add(run)
            prompt = VisibilityPrompt(
                run_id=run_id,
                prompt_key="u-1",
                intent_class="I",
                role="unbranded_discovery",
                text="Какой сервис выбрать?",
                rationale="Сохранённый сценарий",
                sequence=1,
            )
            session.add(prompt)
            await session.flush()
            session.add_all(
                [
                    RunArtifact(
                        run_id=run_id,
                        stage_key="report",
                        artifact_key="final_report",
                        status="completed",
                        model="test/report-model",
                        output_json={"headline": "Сохранённый отчёт"},
                        raw_text='{"headline":"Сохранённый отчёт"}',
                        usage_json={"total_tokens": 321},
                    ),
                    ModelAnswer(
                        run_id=run_id,
                        prompt_id=prompt.id,
                        provider_key="openai",
                        model="test/model",
                        mode="web",
                        status="completed",
                        response_text="Неизменяемый сырой ответ.",
                        citations_json=[{"url": "https://source.example/evidence"}],
                        usage_json={"total_tokens": 123},
                    ),
                ]
            )
            await session.commit()

        async def evidence_snapshot() -> bytes:
            async with SessionLocal() as session:
                saved_run = (
                    await session.execute(select(Run).where(Run.id == run_id))
                ).scalar_one()
                artifacts = list(
                    (
                        await session.execute(
                            select(RunArtifact)
                            .where(RunArtifact.run_id == run_id)
                            .order_by(RunArtifact.id)
                        )
                    )
                    .scalars()
                    .all()
                )
                prompts = list(
                    (
                        await session.execute(
                            select(VisibilityPrompt)
                            .where(VisibilityPrompt.run_id == run_id)
                            .order_by(VisibilityPrompt.id)
                        )
                    )
                    .scalars()
                    .all()
                )
                answers = list(
                    (
                        await session.execute(
                            select(ModelAnswer)
                            .where(ModelAnswer.run_id == run_id)
                            .order_by(ModelAnswer.id)
                        )
                    )
                    .scalars()
                    .all()
                )
            snapshot = {
                "run": {
                    "status": saved_run.status.value,
                    "config": saved_run.config_json,
                    "progress": [
                        saved_run.progress_current,
                        saved_run.progress_total,
                        saved_run.progress_percent,
                    ],
                    "stage": [
                        saved_run.stage_key,
                        saved_run.stage_label,
                        saved_run.stage_detail,
                    ],
                    "state_revision": saved_run.state_revision,
                    "resume_count": saved_run.resume_count,
                    "resume_reason": saved_run.resume_reason,
                    "analysis_markdown": saved_run.analysis_markdown,
                    "report_json": saved_run.report_json,
                },
                "artifacts": [
                    {
                        "stage_key": artifact.stage_key,
                        "artifact_key": artifact.artifact_key,
                        "status": artifact.status,
                        "model": artifact.model,
                        "output_json": artifact.output_json,
                        "raw_text": artifact.raw_text,
                        "usage_json": artifact.usage_json,
                    }
                    for artifact in artifacts
                ],
                "prompts": [
                    {
                        "prompt_key": saved_prompt.prompt_key,
                        "intent_class": saved_prompt.intent_class,
                        "role": saved_prompt.role,
                        "text": saved_prompt.text,
                        "rationale": saved_prompt.rationale,
                        "sequence": saved_prompt.sequence,
                    }
                    for saved_prompt in prompts
                ],
                "answers": [
                    {
                        "provider_key": answer.provider_key,
                        "model": answer.model,
                        "mode": answer.mode,
                        "status": answer.status,
                        "response_text": answer.response_text,
                        "citations_json": answer.citations_json,
                        "usage_json": answer.usage_json,
                        "error_message": answer.error_message,
                    }
                    for answer in answers
                ],
            }
            return json.dumps(
                snapshot,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")

        try:
            before = await evidence_snapshot()
            with (
                patch("app.routes.runs.coordinator.wake") as wake,
                patch("app.routes.runs.bus.reset") as reset_bus,
                patch(
                    "app.routes.runs.pending_run_count",
                    new_callable=AsyncMock,
                ) as pending_count,
            ):
                response = await self.client.post(f"/api/runs/{run_id}/retry")
            after = await evidence_snapshot()

            self.assertEqual(response.status_code, 409)
            self.assertEqual(
                response.json()["detail"],
                "Готовую проверку нельзя перезапустить.",
            )
            self.assertEqual(after, before)
            wake.assert_not_called()
            reset_bus.assert_not_called()
            pending_count.assert_not_awaited()
        finally:
            async with SessionLocal() as session:
                await session.execute(delete(Run).where(Run.id == run_id))
                await session.commit()

    async def test_operator_review_run_cannot_enter_automatic_retry(self) -> None:
        run_id = f"test-review-retry-{uuid.uuid4()}"
        async with SessionLocal() as session:
            session.add(
                Run(
                    id=run_id,
                    domain="review.example.com",
                    status=RunStatus.failed,
                    stage_key="source_review_required",
                    stage_label="Нужна проверка источников",
                    stage_detail="Каталог не подтверждён.",
                    error_message="Каталог не подтверждён.",
                    config_json={"pipeline_version": "immutable-test"},
                    state_revision=7,
                )
            )
            await session.commit()

        async def snapshot() -> tuple[object, ...]:
            async with SessionLocal() as session:
                return (
                    await session.execute(
                        select(
                            Run.status,
                            Run.stage_key,
                            Run.stage_label,
                            Run.stage_detail,
                            Run.error_message,
                            Run.resume_count,
                            Run.state_revision,
                        ).where(Run.id == run_id)
                    )
                ).one()

        try:
            before = await snapshot()
            with (
                patch("app.routes.runs.coordinator.wake") as wake,
                patch("app.routes.runs.bus.reset") as reset_bus,
                patch(
                    "app.routes.runs.pending_run_count",
                    new_callable=AsyncMock,
                ) as pending_count,
            ):
                response = await self.client.post(f"/api/runs/{run_id}/retry")
            after = await snapshot()

            self.assertEqual(response.status_code, 409)
            self.assertIn(
                "нельзя безопасно продолжить автоматически", response.json()["detail"]
            )
            self.assertEqual(after, before)
            wake.assert_not_called()
            reset_bus.assert_not_called()
            pending_count.assert_not_awaited()
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

    async def test_public_history_uses_stable_bounded_cursor_pages(self) -> None:
        first_id = f"cursor-a-{uuid.uuid4()}"
        second_id = f"cursor-b-{uuid.uuid4()}"
        first_created = datetime(2099, 1, 1, tzinfo=timezone.utc)
        second_created = first_created + timedelta(seconds=1)
        async with SessionLocal() as session:
            session.add_all(
                [
                    Run(
                        id=first_id,
                        domain="cursor-first.example",
                        status=RunStatus.completed,
                        config_json={},
                        created_at=first_created,
                        report_json={"large": "x" * 100_000},
                    ),
                    Run(
                        id=second_id,
                        domain="cursor-second.example",
                        status=RunStatus.completed,
                        config_json={},
                        created_at=second_created,
                        report_json={"large": "y" * 100_000},
                    ),
                ]
            )
            await session.commit()
        try:
            first_page = await self.client.get("/api/runs", params={"limit": 1})
            self.assertEqual(first_page.status_code, 200)
            self.assertEqual([row["id"] for row in first_page.json()], [second_id])
            cursor = first_page.json()[0]
            second_page = await self.client.get(
                "/api/runs",
                params={
                    "limit": 1,
                    "before_created_at": cursor["created_at"],
                    "before_id": cursor["id"],
                },
            )
            self.assertEqual(second_page.status_code, 200)
            self.assertEqual([row["id"] for row in second_page.json()], [first_id])
            self.assertNotIn("report_json", first_page.text)
            self.assertLess(len(first_page.content), 2_000)
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

    async def test_public_illustrations_come_only_from_report_snapshot(self) -> None:
        run_id = f"test-public-illustrations-{uuid.uuid4()}"
        share_token = f"share-{uuid.uuid4().hex}"
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        static_dir = Path(temporary_directory.name) / "static"
        generated_dir = static_dir / "generated"
        run_dir = generated_dir / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "01.png").write_bytes(b"canonical-image")
        static_patch = patch(
            "app.services.publication_contract.STATIC_DIR",
            static_dir,
        )
        generated_patch = patch(
            "app.services.publication_contract.GENERATED_DIR",
            generated_dir,
        )
        static_patch.start()
        generated_patch.start()
        self.addCleanup(static_patch.stop)
        self.addCleanup(generated_patch.stop)
        canonical = {
            "sequence": 1,
            "title": "Актуальный вывод",
            "caption": "Подпись из канонического снимка отчёта.",
            "alt_text": "Актуальное описание иллюстрации.",
            "file_url": f"/static/generated/{run_id}/01.png",
        }
        stale_marker = "УСТАРЕВШАЯ ORM-ПОДПИСЬ"
        async with SessionLocal() as session:
            session.add(
                Run(
                    id=run_id,
                    domain="illustrations.example.com",
                    status=RunStatus.completed,
                    config_json={},
                    progress_current=100,
                    progress_total=100,
                    progress_percent=100,
                    share_token=share_token,
                    report_json={"illustrations": [canonical]},
                )
            )
            await session.flush()
            session.add(
                ReportIllustration(
                    run_id=run_id,
                    sequence=1,
                    title=stale_marker,
                    caption="Старые расчёты из ORM.",
                    alt_text="Старое описание из ORM.",
                    file_url="/static/generated/stale/01.png",
                    generation_prompt="old-generation-prompt",
                    model="test/image-model",
                )
            )
            await session.commit()

        try:
            detail = await self.client.get(f"/api/runs/{run_id}")
            shared = await self.client.get(f"/api/shared/{share_token}")
            history = await self.client.get("/api/runs")

            self.assertEqual(detail.status_code, 200)
            self.assertEqual(shared.status_code, 200)
            self.assertEqual(history.status_code, 200)
            self.assertEqual(detail.json()["illustrations"], [canonical])
            self.assertEqual(shared.json()["illustrations"], [canonical])
            self.assertNotIn(stale_marker, detail.text)
            self.assertNotIn(stale_marker, shared.text)

            history_row = next(item for item in history.json() if item["id"] == run_id)
            self.assertNotIn("illustrations", history_row)
            self.assertNotIn(stale_marker, history.text)
        finally:
            async with SessionLocal() as session:
                await session.execute(
                    delete(ReportIllustration).where(
                        ReportIllustration.run_id == run_id
                    )
                )
                await session.execute(delete(Run).where(Run.id == run_id))
                await session.commit()

    async def test_public_detail_never_falls_back_to_orm_illustrations(self) -> None:
        run_id = f"test-public-empty-illustrations-{uuid.uuid4()}"
        stale_marker = "УСТАРЕВШИЙ ORM-ALT"
        async with SessionLocal() as session:
            session.add(
                Run(
                    id=run_id,
                    domain="empty-illustrations.example.com",
                    status=RunStatus.completed,
                    config_json={},
                    progress_current=100,
                    progress_total=100,
                    progress_percent=100,
                    report_json={"illustrations": []},
                )
            )
            await session.flush()
            session.add(
                ReportIllustration(
                    run_id=run_id,
                    sequence=1,
                    title="Старый заголовок",
                    caption="Старая подпись.",
                    alt_text=stale_marker,
                    file_url="/static/generated/stale/01.png",
                    generation_prompt="old-generation-prompt",
                    model="test/image-model",
                )
            )
            await session.commit()

        try:
            response = await self.client.get(f"/api/runs/{run_id}")

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["illustrations"], [])
            self.assertNotIn(stale_marker, response.text)
        finally:
            async with SessionLocal() as session:
                await session.execute(
                    delete(ReportIllustration).where(
                        ReportIllustration.run_id == run_id
                    )
                )
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
