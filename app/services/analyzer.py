"""Multi-stage AI visibility analysis.

LLMs annotate evidence and write explanations. All public metrics are computed
deterministically from those atomic annotations; missing data never becomes a
zero.
"""
from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import json
import logging
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from sqlalchemy import delete, select, update
from sqlalchemy.orm.attributes import flag_modified

from app.config import settings
from app.db import SessionLocal
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
from app.services.openrouter import (
    WEB_ATTESTATION_VERSION,
    ImageResult,
    OpenRouterError,
    OpenRouterPolicyError,
    PanelModel,
    WebSearchPolicy,
    chat,
    generate_image,
    panel_models,
    web_request_policy,
)
from app.services.analysis_critic import (
    CRITIC_MODEL,
    CRITIC_VERSION,
    MAX_CRITIC_ITERATIONS,
    repair_analysis_review,
    review_analysis,
)
from app.services.report_semantic_gate import (
    CANONICAL_OBSERVATIONAL_MEMORY_LIMITATION,
    CANONICAL_UNAVAILABLE_PORTFOLIO_LIMITATION,
    MAX_FINAL_REPORT_REPAIRS,
    REPORT_SEMANTIC_GATE_VERSION,
    REPORT_SEMANTIC_MODEL,
    deterministic_report_semantic_errors,
    metric_availability_contract,
    normalize_report_semantic_review,
    report_semantic_blockers,
    review_final_report_semantics,
    validate_report_semantic_review,
)
from app.services.progress import complete_run, fail_run, update_progress
from app.services.robots_parser import robots_path_allowed
from app.services.run_lease import (
    RunLeaseLostError,
    assert_run_lease,
    lease_owner_for,
)
from app.services.site_preview import get_saved_site_preview

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"
GENERATED_DIR = STATIC_DIR / "generated"
PROMPT_VERSION = "aiv-2026-07-30-system-v2"
MARKET_RESEARCH_VERSION = f"{PROMPT_VERSION}-market-research-v2"
PROMPT_SET_VERSION = f"{PROMPT_VERSION}-intent-v3"
PROMPT_SET_REVIEW_VERSION = f"{PROMPT_VERSION}-intent-review-v4"
PANEL_CONTRACT_VERSION = f"{PROMPT_VERSION}-panel-v2"
LEGACY_PANEL_CONTRACT_VERSION = f"{PROMPT_VERSION}-panel-v1"
ENTITY_CATALOG_CHUNK_VERSION = f"{PROMPT_VERSION}-entities-v6"
ENTITY_CATALOG_VERSION = f"{PROMPT_VERSION}-entities-v8"
ANNOTATION_VERSION = f"{PROMPT_VERSION}-annotations-v14"
METRICS_VERSION = f"{PROMPT_VERSION}-metrics-v17"
ANALYSIS_CRITIC_VERSION = f"{PROMPT_VERSION}-{CRITIC_VERSION}"
TECHNICAL_REVIEW_VERSION = f"{PROMPT_VERSION}-technical-v3"
FINAL_REPORT_VERSION = f"{PROMPT_VERSION}-final-v17"
FINAL_CORPUS_MANIFEST_VERSION = f"{PROMPT_VERSION}-final-corpus-v3"
FINAL_CONTEXT_SELECTION_VERSION = f"{PROMPT_VERSION}-final-selection-v4"
FINAL_CONTEXT_MAX_ANSWERS = 12
FINAL_REPAIR_TOKEN_RESERVE = 45_000
FINAL_REPORT_AUTHOR_ARTIFACT_KEY = "final_report_author_candidate"
MAX_FINAL_STRUCTURE_REPAIRS = 1
LEGACY_PANEL_EVIDENCE_VERSION = "aiv-legacy-panel-evidence-v2"
LEGACY_MEMORY_OBSERVATION_REASON = "legacy_memory_request_not_enforced"
LEGACY_PANEL_MODELS = {
    "openai": "openai/gpt-chat-latest",
    "gemini": "google/gemini-3.6-flash",
    "perplexity": "perplexity/sonar-pro-search",
    "deepseek": "deepseek/deepseek-v4-pro",
    "claude": "anthropic/claude-sonnet-5",
}
LEGACY_MEMORY_MODELS = {
    key: model
    for key, model in LEGACY_PANEL_MODELS.items()
    if key != "perplexity"
}
LEGACY_PIPELINE_VERSION = "aiv-2026-07"
LEGACY_PROMPT_SET_VERSIONS = {
    "aiv-2026-07-30",
    "aiv-2026-07-30-system-v2",
    "aiv-2026-07-30-system-v2-intent-v2",
}
ILLUSTRATION_CONCEPTS_VERSION = f"{PROMPT_VERSION}-visual-v13"
ILLUSTRATION_COPY_FALLBACK_VERSION = f"{PROMPT_VERSION}-visual-fallback-v1"
ILLUSTRATION_GENERATION_VERSION = f"{PROMPT_VERSION}-image-gen-v2"
ILLUSTRATION_QA_VERSION = f"{PROMPT_VERSION}-image-qa-v8"
ILLUSTRATION_QUALITY_ATTEMPTS = 2
ILLUSTRATION_MAX_ATTEMPTS = 3
ILLUSTRATION_ROLE_CONCURRENCY = 2
ANALYSIS_MODEL = settings.OPENROUTER_ANALYSIS_MODEL or settings.OPENROUTER_MODEL
PROCESSING_MODEL = settings.OPENROUTER_PROCESSING_MODEL
ILLUSTRATION_CONCEPT_MODEL = (
    settings.OPENROUTER_ILLUSTRATION_CONCEPT_MODEL or ANALYSIS_MODEL
)
PROCESSING_BATCH_CONCURRENCY = 3
ANNOTATION_COMPLETION_ATTEMPTS = 2
ANSWER_ANALYSIS_CHAR_LIMIT = 16_000
CRITIC_ANSWER_CHAR_LIMIT = 24_000
ENTITY_CATALOG_CHUNK_CHAR_LIMIT = 48_000
ANNOTATION_BATCH_CHAR_LIMIT = 64_000
_CACHE_UNSET = object()
SITE_PAGE_MANIFEST_KEY = "site_page_manifest"
SITE_PAGE_MANIFEST_VERSION = "aiv-2026-07-30-site-page-manifest-v1"
AI_LABELS = {
    "GPTBot",
    "OAI-SearchBot",
    "ChatGPT-User",
    "ClaudeBot",
    "PerplexityBot",
    "Perplexity-User",
    "Googlebot-desktop",
    "Google-Agent-desktop",
    "DeepSeekBot",
}
CONTROL_LABEL = "Chrome-control"
LEGACY_RESPONSE_BODY_LIMIT_BYTES = 768 * 1024
INTENT_DEFINITIONS: dict[str, str] = {
    "I": "Information Seeking — общие сведения и понимание темы.",
    "E": "Evaluative — сравнение вариантов и критериев выбора.",
    "T": "Transactional — готовность купить, заказать или принять решение.",
    "NB": "Need Based — задача, боль, ограничение или контекст использования.",
    "NAV": "Navigation — источник, площадка, обзор, агрегатор или точка входа.",
    "TR": "Trend-Driven — тренды, новизна, популярность или меняющееся поведение.",
}
_TRANSIENT_HTTP_STATUSES = frozenset({408, 425, 429})
_ENTITY_SCHEMA_TYPES = frozenset(
    {
        "article",
        "blogposting",
        "brand",
        "book",
        "corporation",
        "course",
        "dataset",
        "event",
        "financialproduct",
        "jobposting",
        "localbusiness",
        "medicalorganization",
        "mobileapplication",
        "newsarticle",
        "offer",
        "organization",
        "place",
        "product",
        "productmodel",
        "report",
        "service",
        "softwareapplication",
        "touristattraction",
        "touristtrip",
        "trip",
        "vehicle",
        "webapplication",
    }
)

LIVE_RUSSIAN_RULES = """
Пиши по-русски для руководителя маркетинга или продукта.
Называй действующее лицо: сайт отдаёт, модель называет, пользователь выбирает.
Каждое число связывай с носителем и смыслом. Не выдавай неизвестность за ноль.
Начинай раздел с вывода, затем показывай доказательство и действие.
Используй живые глаголы, короткие абзацы и разные по длине предложения.
Не используй канцелярит, пассивные конструкции, рекламные эпитеты,
«важно отметить», «в современном мире», «комплексный подход» и ложные
противопоставления вида «это не X, а Y».
Заголовок должен сообщать конкретный вывод и читаться как законченное
предложение. Не превращай отчёт в набор слоганов.
Русская типографика: кавычки «ёлочки», длинное тире, знак №, неразрывные
пробелы между числом и единицей.
""".strip()


SITE_PROFILE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "brand_name": {"type": "string"},
        "brand_aliases": {"type": "array", "items": {"type": "string"}},
        "site_type": {"type": "string"},
        "category": {"type": "string"},
        "topics": {"type": "array", "items": {"type": "string"}},
        "market": {"type": "string"},
        "business_model": {"type": "string"},
        "products": {"type": "array", "items": {"type": "string"}},
        "audiences": {"type": "array", "items": {"type": "string"}},
        "customer_jobs": {"type": "array", "items": {"type": "string"}},
        "decision_criteria": {"type": "array", "items": {"type": "string"}},
        "geography": {"type": "array", "items": {"type": "string"}},
        "language": {"type": "string"},
        "positioning": {"type": "string"},
        "entity_scope": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "canonical_name": {"type": "string"},
                    "aliases": {"type": "array", "items": {"type": "string"}},
                    "entity_type": {
                        "type": "string",
                        "enum": [
                            "primary_brand",
                            "business_unit",
                            "product",
                            "service",
                            "platform",
                        ],
                    },
                    "relationship": {
                        "type": "string",
                        "enum": [
                            "self",
                            "owned_by",
                            "operated_by",
                            "offered_by",
                            "unclear",
                        ],
                    },
                    "commercially_relevant": {"type": "boolean"},
                    "evidence": {"type": "string"},
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                    },
                },
                "required": [
                    "canonical_name",
                    "aliases",
                    "entity_type",
                    "relationship",
                    "commercially_relevant",
                    "evidence",
                    "confidence",
                ],
            },
        },
        "evidence": {"type": "array", "items": {"type": "string"}},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": [
        "brand_name",
        "brand_aliases",
        "site_type",
        "category",
        "topics",
        "market",
        "business_model",
        "products",
        "audiences",
        "customer_jobs",
        "decision_criteria",
        "geography",
        "language",
        "positioning",
        "entity_scope",
        "evidence",
        "uncertainties",
        "confidence",
    ],
}

MARKET_RESEARCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "status": {
            "type": "string",
            "enum": ["ready", "limited", "blocked"],
        },
        "site_confirmed": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "primary_brand": {"type": "string"},
                "brand_aliases": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "site_type": {"type": "string"},
                "category": {"type": "string"},
                "products": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "topics": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "geography": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "evidence": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "claim": {"type": "string"},
                            "url": {"type": "string"},
                            "excerpt": {"type": "string"},
                        },
                        "required": ["claim", "url", "excerpt"],
                    },
                },
            },
            "required": [
                "primary_brand",
                "brand_aliases",
                "site_type",
                "category",
                "products",
                "topics",
                "geography",
                "evidence",
            ],
        },
        "external_market_research": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "market": {"type": "string"},
                "topics": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "geography": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "audiences": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "customer_jobs": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "decision_criteria": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "terminology": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "term": {"type": "string"},
                            "meaning": {"type": "string"},
                        },
                        "required": ["term", "meaning"],
                    },
                },
                "evidence": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "dimension": {
                                "type": "string",
                                "enum": [
                                    "market",
                                    "topics",
                                    "geography",
                                    "audiences",
                                    "customer_jobs",
                                    "decision_criteria",
                                    "terminology",
                                ],
                            },
                            "claim": {"type": "string"},
                            "source_urls": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "evidence": {"type": "string"},
                            "confidence": {
                                "type": "string",
                                "enum": ["high", "medium", "low"],
                            },
                        },
                        "required": [
                            "dimension",
                            "claim",
                            "source_urls",
                            "evidence",
                            "confidence",
                        ],
                    },
                },
            },
            "required": [
                "market",
                "topics",
                "geography",
                "audiences",
                "customer_jobs",
                "decision_criteria",
                "terminology",
                "evidence",
            ],
        },
        "sources": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "url": {"type": "string"},
                    "title": {"type": "string"},
                    "publisher": {"type": "string"},
                    "evidence": {"type": "string"},
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                    },
                },
                "required": [
                    "url",
                    "title",
                    "publisher",
                    "evidence",
                    "confidence",
                ],
            },
        },
        "uncertainties": {
            "type": "array",
            "items": {"type": "string"},
        },
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low"],
        },
    },
    "required": [
        "status",
        "site_confirmed",
        "external_market_research",
        "sources",
        "uncertainties",
        "confidence",
    ],
}

PROMPT_SET_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "prompts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "prompt_key": {"type": "string"},
                    "intent_class": {
                        "type": "string",
                        "enum": ["I", "E", "T", "NB", "NAV", "TR"],
                    },
                    "role": {
                        "type": "string",
                        "enum": ["unbranded_discovery", "brand_diagnostic"],
                    },
                    "text": {"type": "string"},
                    "rationale": {"type": "string"},
                    "choice_request": {"type": "boolean"},
                },
                "required": [
                    "prompt_key",
                    "intent_class",
                    "role",
                    "text",
                    "rationale",
                    "choice_request",
                ],
            },
        }
    },
    "required": ["prompts"],
}

PROMPT_SET_REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["pass", "revise"],
        },
        "summary": {"type": "string"},
        "checks": {
            "type": "array",
            # Ровно шесть: flash-критик без этого ограничения возвращал одну
            # проверку с verdict=pass (прогон 5ae13350, 2026-08-21).
            "minItems": 6,
            "maxItems": 6,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "prompt_key": {"type": "string"},
                    "declared_intent": {
                        "type": "string",
                        "enum": ["I", "E", "T", "NB", "NAV", "TR"],
                    },
                    "dominant_intent": {
                        "type": "string",
                        "enum": ["I", "E", "T", "NB", "NAV", "TR"],
                    },
                    "matches": {"type": "boolean"},
                    "grounded_in_research": {"type": "boolean"},
                    "supporting_evidence": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "unsupported_assumptions": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "reason": {"type": "string"},
                    "fix_instruction": {"type": "string"},
                },
                "required": [
                    "prompt_key",
                    "declared_intent",
                    "dominant_intent",
                    "matches",
                    "grounded_in_research",
                    "supporting_evidence",
                    "unsupported_assumptions",
                    "reason",
                    "fix_instruction",
                ],
            },
        },
    },
    "required": ["verdict", "summary", "checks"],
}

TECHNICAL_REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "overall_conclusion": {"type": "string"},
        "render_conclusion": {"type": "string"},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "severity": {
                        "type": "string",
                        "enum": ["critical", "important", "observation"],
                    },
                    "title": {"type": "string"},
                    "evidence": {"type": "string"},
                    "business_effect": {"type": "string"},
                    "action": {"type": "string"},
                },
                "required": [
                    "severity",
                    "title",
                    "evidence",
                    "business_effect",
                    "action",
                ],
            },
        },
        "limitations": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "overall_conclusion",
        "render_conclusion",
        "findings",
        "limitations",
    ],
}

ENTITY_CATALOG_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "target_aliases": {"type": "array", "items": {"type": "string"}},
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "canonical_name": {"type": "string"},
                    "aliases": {
                        "type": "array",
                        "items": {
                            "anyOf": [
                                {"type": "string"},
                                {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "value": {"type": "string"},
                                        "match_policy": {
                                            "type": "string",
                                            "enum": [
                                                "standalone",
                                                "requires_target_attribution",
                                            ],
                                        },
                                    },
                                    "required": ["value", "match_policy"],
                                },
                            ]
                        },
                    },
                    "category": {
                        "type": "string",
                        "enum": ["target", "competitor", "other"],
                    },
                    "target_relationship": {
                        "type": "string",
                        "enum": [
                            "exact_target",
                            "portfolio_entity",
                            "competitor",
                            "unrelated",
                            "unclear",
                        ],
                    },
                    "commercially_relevant": {"type": "boolean"},
                    "mention_policy": {
                        "type": "string",
                        "enum": [
                            "standalone",
                            "requires_target_attribution",
                        ],
                    },
                    "evidence": {"type": "string"},
                },
                "required": [
                    "canonical_name",
                    "aliases",
                    "category",
                    "target_relationship",
                    "commercially_relevant",
                    "mention_policy",
                    "evidence",
                ],
            },
        },
        "uncertainties": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["target_aliases", "entities", "uncertainties"],
}

ANNOTATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "answers": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "answer_id": {"type": "integer"},
                    "valid": {"type": "boolean"},
                    "target_mentioned": {"type": "boolean"},
                    "target_position": {
                        "anyOf": [{"type": "integer", "minimum": 1}, {"type": "null"}]
                    },
                    "target_role": {
                        "type": "string",
                        "enum": [
                            "recommended",
                            "conditional",
                            "mentioned",
                            "excluded",
                            "absent",
                            "unknown",
                        ],
                    },
                    "sentiment": {
                        "type": "string",
                        "enum": ["positive", "neutral", "mixed", "negative", "unknown"],
                    },
                    "entity_mentions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "canonical_name": {"type": "string"},
                                "position": {
                                    "anyOf": [
                                        {"type": "integer", "minimum": 1},
                                        {"type": "null"},
                                    ]
                                },
                                "role": {
                                    "type": "string",
                                    "enum": [
                                        "recommended",
                                        "conditional",
                                        "mentioned",
                                        "excluded",
                                    ],
                                },
                                "attributed_to_target": {"type": "boolean"},
                                "evidence": {
                                    "type": "string",
                                    "description": (
                                        "One exact contiguous substring copied "
                                        "from the raw answer. Preserve case, "
                                        "spelling, punctuation, Markdown and "
                                        "line breaks; never join distant spans."
                                    ),
                                },
                            },
                            "required": [
                                "canonical_name",
                                "position",
                                "role",
                                "attributed_to_target",
                                "evidence",
                            ],
                        },
                    },
                    "brand_answer": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "directness": {
                                "type": "string",
                                "enum": [
                                    "direct",
                                    "partial",
                                    "evasive",
                                    "refusal",
                                    "not_applicable",
                                ],
                            },
                            "specificity": {
                                "type": "string",
                                "enum": [
                                    "specific",
                                    "generic",
                                    "none",
                                    "contradictory",
                                    "not_applicable",
                                ],
                            },
                            "supported_facets": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": [
                                        "identity",
                                        "offering",
                                        "portfolio",
                                        "market",
                                        "differentiation",
                                        "proof",
                                        "reputation",
                                        "limitations",
                                    ],
                                },
                            },
                            "contradictions": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": [
                            "directness",
                            "specificity",
                            "supported_facets",
                            "contradictions",
                        ],
                    },
                    "evidence": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "description": (
                                "Exact contiguous raw-answer substring; no "
                                "paraphrase, corrected spelling or added quotes."
                            ),
                        },
                    },
                    "uncertainties": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "answer_id",
                    "valid",
                    "target_mentioned",
                    "target_position",
                    "target_role",
                    "sentiment",
                    "entity_mentions",
                    "brand_answer",
                    "evidence",
                    "uncertainties",
                ],
            },
        }
    },
    "required": ["answers"],
}

FINAL_REPORT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "headline": {"type": "string"},
        "headline_emphasis": {
            "type": "array",
            "items": {"type": "string"},
        },
        "verdict": {"type": "string"},
        "executive_summary": {"type": "string"},
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "heading": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["heading", "body"],
            },
        },
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "priority": {
                        "type": "string",
                        "enum": ["now", "next", "later"],
                    },
                    "title": {"type": "string"},
                    "why": {"type": "string"},
                    "step": {"type": "string"},
                    "evidence": {"type": "string"},
                },
                "required": ["priority", "title", "why", "step", "evidence"],
            },
        },
        "limitations": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "headline",
        "headline_emphasis",
        "verdict",
        "executive_summary",
        "sections",
        "actions",
        "limitations",
    ],
}

ILLUSTRATION_CONCEPTS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "illustrations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "role": {
                        "type": "string",
                        "enum": [
                            "technical_access",
                            "competitive_visibility",
                            "web_memory_gap",
                        ],
                    },
                    "title": {"type": "string"},
                    "caption": {"type": "string"},
                    "alt_text": {"type": "string"},
                    "core_claim": {"type": "string"},
                    "evidence_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "context_for_image": {"type": "string"},
                    "creative_brief": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "visual_thesis": {"type": "string"},
                            "scene": {"type": "string"},
                            "composition": {"type": "string"},
                            "materials_and_light": {"type": "string"},
                            "emotional_tone": {"type": "string"},
                            "target_treatment": {"type": "string"},
                            "diversity_move": {"type": "string"},
                        },
                        "required": [
                            "visual_thesis",
                            "scene",
                            "composition",
                            "materials_and_light",
                            "emotional_tone",
                            "target_treatment",
                            "diversity_move",
                        ],
                    },
                },
                "required": [
                    "role",
                    "title",
                    "caption",
                    "alt_text",
                    "core_claim",
                    "evidence_paths",
                    "context_for_image",
                    "creative_brief",
                ],
            },
        },
    },
    "required": ["illustrations"],
}

ILLUSTRATION_QA_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "usable": {"type": "boolean"},
        "inferred_message": {"type": "string"},
        "facts_grounded": {"type": "boolean"},
        "claim_readable": {"type": "boolean"},
        "unsupported_assertions": {"type": "array", "items": {"type": "string"}},
        "visible_text_problems": {"type": "array", "items": {"type": "string"}},
        "hard_blockers": {"type": "array", "items": {"type": "string"}},
        "scores": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "context_specificity": {"type": "integer", "minimum": 1, "maximum": 5},
                "visual_story": {"type": "integer", "minimum": 1, "maximum": 5},
                "distinctiveness": {"type": "integer", "minimum": 1, "maximum": 5},
                "hierarchy": {"type": "integer", "minimum": 1, "maximum": 5},
                "craft": {"type": "integer", "minimum": 1, "maximum": 5},
                "richness": {"type": "integer", "minimum": 1, "maximum": 5},
            },
            "required": [
                "context_specificity",
                "visual_story",
                "distinctiveness",
                "hierarchy",
                "craft",
                "richness",
            ],
        },
        "strengths": {"type": "array", "items": {"type": "string"}},
        "improvements": {"type": "array", "items": {"type": "string"}},
        "retry_instruction": {"type": "string"},
    },
    "required": [
        "usable",
        "inferred_message",
        "facts_grounded",
        "claim_readable",
        "unsupported_assertions",
        "visible_text_problems",
        "hard_blockers",
        "scores",
        "strengths",
        "improvements",
        "retry_instruction",
    ],
}


def _artifact_cache_matches(
    artifact: RunArtifact,
    *,
    input_json: dict[str, Any] | list[Any] | None | object = _CACHE_UNSET,
    model: str | None | object = _CACHE_UNSET,
    prompt_version: str = PROMPT_VERSION,
) -> bool:
    if (
        artifact.status != "completed"
        or artifact.prompt_version != prompt_version
        or artifact.output_json is None
    ):
        return False
    if input_json is not _CACHE_UNSET and artifact.input_json != input_json:
        return False
    if model is not _CACHE_UNSET and artifact.model != model:
        return False
    return True


async def _artifact_output(
    run_id: str,
    artifact_key: str,
    *,
    input_json: dict[str, Any] | list[Any] | None | object = _CACHE_UNSET,
    model: str | None | object = _CACHE_UNSET,
    prompt_version: str = PROMPT_VERSION,
) -> dict[str, Any] | list[Any] | None:
    async with SessionLocal() as session:
        artifact = (
            await session.execute(
                select(RunArtifact).where(
                    RunArtifact.run_id == run_id,
                    RunArtifact.artifact_key == artifact_key,
                )
            )
        ).scalar_one_or_none()
        if artifact is not None and _artifact_cache_matches(
            artifact,
            input_json=input_json,
            model=model,
            prompt_version=prompt_version,
        ):
            return artifact.output_json
    return None


async def _save_artifact(
    run_id: str,
    *,
    stage_key: str,
    artifact_key: str,
    status: str,
    model: str | None = None,
    input_json: dict[str, Any] | list[Any] | None = None,
    output_json: dict[str, Any] | list[Any] | None = None,
    raw_text: str | None = None,
    usage_json: dict[str, Any] | None = None,
    error_message: str | None = None,
    prompt_version: str = PROMPT_VERSION,
) -> None:
    await assert_run_lease(run_id)
    async with SessionLocal() as session:
        artifact = (
            await session.execute(
                select(RunArtifact).where(
                    RunArtifact.run_id == run_id,
                    RunArtifact.artifact_key == artifact_key,
                )
            )
        ).scalar_one_or_none()
        if artifact is None:
            artifact = RunArtifact(
                run_id=run_id,
                stage_key=stage_key,
                artifact_key=artifact_key,
            )
            session.add(artifact)
        artifact.stage_key = stage_key
        artifact.status = status
        artifact.model = model
        artifact.prompt_version = prompt_version
        artifact.input_json = input_json
        artifact.output_json = output_json
        artifact.raw_text = raw_text
        artifact.usage_json = usage_json
        artifact.error_message = error_message[:1000] if error_message else None
        await session.commit()


async def _structured_artifact(
    run_id: str,
    *,
    stage_key: str,
    artifact_key: str,
    schema: dict[str, Any],
    schema_name: str,
    system: str,
    user_payload: dict[str, Any] | list[Any],
    max_tokens: int = 12_000,
    model: str = ANALYSIS_MODEL,
    reasoning_effort: str = "high",
    prompt_version: str = PROMPT_VERSION,
) -> dict[str, Any]:
    cached = await _artifact_output(
        run_id,
        artifact_key,
        input_json=user_payload,
        model=model,
        prompt_version=prompt_version,
    )
    if isinstance(cached, dict):
        return cached
    await _save_artifact(
        run_id,
        stage_key=stage_key,
        artifact_key=artifact_key,
        status="running",
        model=model,
        input_json=user_payload,
        prompt_version=prompt_version,
    )
    try:
        result = await chat(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False),
                },
            ],
            response_schema=schema,
            schema_name=schema_name,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
            temperature=0.15,
        )
        if not isinstance(result.parsed, dict):
            raise OpenRouterError("Structured response is not an object")
        await _save_artifact(
            run_id,
            stage_key=stage_key,
            artifact_key=artifact_key,
            status="completed",
            model=model,
            input_json=user_payload,
            output_json=result.parsed,
            raw_text=result.text,
            usage_json=result.usage,
            prompt_version=prompt_version,
        )
        return result.parsed
    except Exception as exc:
        await _save_artifact(
            run_id,
            stage_key=stage_key,
            artifact_key=artifact_key,
            status="failed",
            model=model,
            input_json=user_payload,
            error_message=str(exc),
            prompt_version=prompt_version,
        )
        raise


async def _processing_artifact(
    run_id: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run repeatable answer processing on the faster, lower-cost model."""

    return await _structured_artifact(
        run_id,
        **kwargs,
        model=PROCESSING_MODEL,
        reasoning_effort="high",
    )


async def _site_context(run_id: str) -> dict[str, Any]:
    async with SessionLocal() as session:
        domain = (
            await session.execute(select(Run.domain).where(Run.id == run_id))
        ).scalar_one()
        pages = list(
            (
                await session.execute(
                    select(SitePage)
                    .where(SitePage.run_id == run_id)
                    .order_by(SitePage.id)
                )
            )
            .scalars()
            .all()
        )
    return {
        "requested_site": {
            "domain": domain,
            "url": f"https://{domain}/",
        },
        "pages": [
            {
                "url": page.url,
                "page_kind": page.page_kind,
                "title": page.title,
                "meta_description": page.meta_description,
                "main_text": (page.main_text or "")[:16_000],
                "text_length": page.text_length,
                "signals": page.content_signals or {},
            }
            for page in pages
        ]
    }


def _probe_access_outcome(
    probe: DomainProbe,
    baseline_length: int | None = None,
) -> str:
    if probe.error_class or probe.http_status is None:
        return "unknown"
    if probe.challenge_detected:
        return "blocked"
    if (
        probe.http_status in _TRANSIENT_HTTP_STATUSES
        or 500 <= probe.http_status < 600
    ):
        return "unknown"
    if not 200 <= probe.http_status < 400:
        return "blocked"
    signals = probe.content_signals or {}
    if any(
        signals.get(key)
        for key in (
            "looks_like_login_wall",
            "looks_like_captcha_page",
            "looks_like_geo_block",
            "looks_like_error_page",
        )
    ):
        return "blocked"
    if (
        signals.get("looks_like_spa_shell")
        or signals.get("looks_like_redirect_shell")
        or signals.get("render_strategy") == "client_rendered_shell"
    ):
        return "blocked"
    length = int(probe.content_extractable_text_length or 0)
    body_truncated = bool(signals.get("_body_truncated")) or (
        getattr(probe, "response_size_bytes", None)
        == LEGACY_RESPONSE_BODY_LIMIT_BYTES
    )
    if body_truncated:
        # A readable prefix proves that this user-agent received page content,
        # but a shortened prefix cannot be compared with a complete control
        # document and must never become a false access block.
        if length >= 180 or not probe.body_looks_empty:
            return "available"
        return "unknown"
    if baseline_length and baseline_length >= 500:
        accessible = length >= max(250, int(baseline_length * 0.25))
    else:
        accessible = length >= 180 or not probe.body_looks_empty
    return "available" if accessible else "blocked"


def _probe_accessible(probe: DomainProbe, baseline_length: int | None = None) -> bool:
    return _probe_access_outcome(probe, baseline_length) == "available"


async def apply_ua_conditional_block(run_id: str) -> int:
    async with SessionLocal() as session:
        probes = list(
            (
                await session.execute(
                    select(DomainProbe)
                    .where(
                        DomainProbe.run_id == run_id,
                        DomainProbe.probe_type == ProbeType.main_page,
                    )
                    .order_by(DomainProbe.id)
                )
            )
            .scalars()
            .all()
        )
        by_url: dict[str, dict[str, DomainProbe]] = defaultdict(dict)
        for probe in probes:
            by_url[probe.target_url][probe.user_agent_label] = probe
        changed = 0
        for variants in by_url.values():
            control = variants.get(CONTROL_LABEL)
            if control is None or not _probe_accessible(control):
                continue
            baseline_length = int(control.content_extractable_text_length or 0)
            for label, probe in variants.items():
                if label not in AI_LABELS or _probe_accessible(probe, baseline_length):
                    continue
                explicit_block = (
                    probe.http_status in {401, 403, 429, 451}
                    or probe.challenge_detected
                    or any(
                        (probe.content_signals or {}).get(key)
                        for key in (
                            "looks_like_login_wall",
                            "looks_like_captcha_page",
                            "looks_like_geo_block",
                            "looks_like_error_page",
                        )
                    )
                )
                if not explicit_block:
                    continue
                markers = list(probe.detected_protections or [])
                if "ua-conditional-block" not in markers:
                    markers.append("ua-conditional-block")
                    probe.detected_protections = markers
                    flag_modified(probe, "detected_protections")
                signals = dict(probe.content_signals or {})
                signals["ua_conditional_block"] = True
                probe.content_signals = signals
                flag_modified(probe, "content_signals")
                changed += 1
        if changed:
            await session.commit()
        return changed


def _family_for_label(label: str) -> str:
    if label in {"GPTBot", "OAI-SearchBot", "ChatGPT-User"}:
        return "OpenAI"
    if label == "ClaudeBot":
        return "Anthropic"
    if label.startswith("Perplexity"):
        return "Perplexity"
    if label.startswith("Google"):
        return "Google"
    if label.startswith("DeepSeek"):
        return "DeepSeek"
    return label


def _is_utility_page(url: str, page_kind: str | None = None) -> bool:
    if page_kind == "utility":
        return True
    path = re.sub(r"[?#].*$", "", url).lower().strip("/")
    segments = {segment for segment in path.split("/") if segment}
    return bool(
        segments.intersection(
            {
                "success",
                "thanks",
                "thank-you",
                "thankyou",
                "submitted",
                "confirmation",
                "confirmed",
            }
        )
    )


def _non_negative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float) and value.is_integer() and value >= 0:
        return int(value)
    return None


def _manifest_page_urls(manifest: dict[str, Any] | None) -> set[str]:
    if not isinstance(manifest, dict):
        return set()
    rows = manifest.get("pages")
    if not isinstance(rows, list):
        return set()
    return {
        str(row.get("url") or "").strip()
        for row in rows
        if isinstance(row, dict) and str(row.get("url") or "").strip()
    }


def _validated_site_page_manifest(
    artifact: RunArtifact | None,
    *,
    run_domain: str,
) -> dict[str, Any] | None:
    """Accept only a manifest that belongs to the requested run domain."""

    if (
        artifact is None
        or artifact.status != "completed"
        or artifact.prompt_version != SITE_PAGE_MANIFEST_VERSION
        or not isinstance(artifact.input_json, dict)
        or not isinstance(artifact.output_json, dict)
    ):
        return None

    normalized_domain = str(run_domain or "").casefold().removeprefix("www.")
    input_domain = (
        str(artifact.input_json.get("domain") or "")
        .casefold()
        .removeprefix("www.")
    )
    selection_limit = artifact.input_json.get("selection_limit")
    output = artifact.output_json
    pages = output.get("pages")
    if (
        not normalized_domain
        or input_domain != normalized_domain
        or type(selection_limit) is not int
        or selection_limit < 1
        or output.get("selection_limit") != selection_limit
        or not isinstance(pages, list)
        or not pages
        or len(pages) > selection_limit
        or output.get("selected_count") != len(pages)
    ):
        return None

    seen: set[str] = set()
    for index, page in enumerate(pages):
        if not isinstance(page, dict):
            return None
        url = page.get("url")
        if not isinstance(url, str) or url in seen:
            return None
        parsed = urlparse(url)
        page_domain = (
            str(parsed.hostname or "").casefold().removeprefix("www.")
        )
        if (
            parsed.scheme not in {"http", "https"}
            or page_domain != normalized_domain
            or not isinstance(page.get("page_kind"), str)
            or (index == 0 and page.get("page_kind") != "home")
        ):
            return None
        seen.add(url)

    discovered = output.get("discovered_count")
    if discovered is not None and (
        type(discovered) is not int or discovered < len(pages)
    ):
        return None
    expected_coverage = (
        "unknown"
        if discovered is None
        else ("complete" if len(pages) >= discovered else "limited")
    )
    if output.get("coverage_state") != expected_coverage:
        return None
    return dict(output)


def _checked_pages_label(count: int) -> str:
    remainder_100 = count % 100
    remainder_10 = count % 10
    if remainder_10 == 1 and remainder_100 != 11:
        return f"{count} проверенная страница"
    if remainder_10 in {2, 3, 4} and remainder_100 not in {12, 13, 14}:
        return f"{count} проверенные страницы"
    return f"{count} проверенных страниц"


def _technical_page_coverage(
    evaluated_pages: int,
    manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build a conservative page-coverage contract for the public report."""

    evaluated = max(0, int(evaluated_pages))
    manifest_urls = _manifest_page_urls(manifest)
    selected = (
        _non_negative_int(manifest.get("selected_count"))
        if isinstance(manifest, dict)
        else None
    )
    if selected is None and manifest_urls:
        selected = len(manifest_urls)

    discovered = (
        _non_negative_int(
            manifest.get("discovered_count", manifest.get("discovered_pages"))
        )
        if isinstance(manifest, dict)
        else None
    )
    required_floor = max(evaluated, selected or 0)
    if discovered is not None and (
        discovered <= 0 or discovered < required_floor
    ):
        # An internally inconsistent denominator must not look like full coverage.
        discovered = None

    declared_state = (
        str(manifest.get("coverage_state") or "").strip().lower()
        if isinstance(manifest, dict)
        else ""
    )
    if discovered is None:
        coverage_rate = None
        coverage_state = "limited" if declared_state == "limited" else "unknown"
    else:
        coverage_rate = round(evaluated / discovered * 100, 1)
        coverage_state = "complete" if evaluated == discovered else "limited"

    coverage_label = {
        "complete": "Проверены все найденные страницы",
        "limited": "Ограниченный срез",
        "unknown": "Охват сайта не определён",
    }[coverage_state]
    scope_label = f"{_checked_pages_label(evaluated)} · {coverage_label.lower()}"
    return {
        "evaluated_pages": evaluated,
        "discovered_pages": discovered,
        "selected_pages": selected,
        "coverage_rate": coverage_rate,
        "coverage_state": coverage_state,
        "coverage_label": coverage_label,
        "scope_label": scope_label,
    }


def _rendering_assessment(
    render_counts: Counter[str],
) -> dict[str, Any]:
    server_readable = sum(
        render_counts[key]
        for key in ("static_html", "server_rendered", "hybrid_ssr_hydration")
    )
    client_only = render_counts["client_rendered_shell"]
    evaluated = server_readable + client_only
    unknown = max(0, sum(render_counts.values()) - evaluated)
    ratio = server_readable / evaluated if evaluated else None

    if client_only and server_readable:
        conclusion = (
            "Часть изученных страниц требует клиентского JavaScript, "
            "остальные отдают основной текст с сервера."
        )
    elif client_only:
        conclusion = (
            "На страницах с определённым способом рендеринга основной текст "
            "появляется только после запуска JavaScript."
        )
    elif server_readable and unknown:
        conclusion = (
            "Подтверждённые содержательные страницы отдают основной текст "
            "с сервера; для части адресов способ рендеринга определить не удалось."
        )
    elif server_readable:
        conclusion = "Основной текст изученных страниц доступен в исходном HTML."
    else:
        conclusion = "Способ рендеринга определить не удалось."

    return {
        "ratio": ratio,
        "evaluated_pages": evaluated,
        "unknown_pages": unknown,
        "conclusion": conclusion,
    }


def _entity_structured_data_types(values: Any) -> list[str]:
    """Separate entity-bearing schema from navigation-only JSON-LD."""

    result: list[str] = []
    for raw_value in values if isinstance(values, list) else []:
        value = str(raw_value or "").strip()
        normalized = re.split(r"[/#:\\]", value)[-1].casefold()
        if (
            normalized in _ENTITY_SCHEMA_TYPES
            or normalized.endswith("organization")
            or normalized.endswith("business")
            or normalized.endswith("article")
            or normalized.endswith("application")
        ):
            result.append(value)
    return list(dict.fromkeys(result))


async def _technical_summary(run_id: str) -> dict[str, Any]:
    async with SessionLocal() as session:
        run = (
            await session.execute(select(Run).where(Run.id == run_id))
        ).scalar_one()
        probes = list(
            (
                await session.execute(
                    select(DomainProbe)
                    .where(DomainProbe.run_id == run_id)
                    .order_by(DomainProbe.id)
                )
            )
            .scalars()
            .all()
        )
        robots = list(
            (
                await session.execute(
                    select(RobotsRule)
                    .where(RobotsRule.run_id == run_id)
                    .order_by(RobotsRule.id)
                )
            )
            .scalars()
            .all()
        )
        pages = list(
            (
                await session.execute(
                    select(SitePage)
                    .where(SitePage.run_id == run_id)
                    .order_by(SitePage.id)
                )
            )
            .scalars()
            .all()
        )
        manifest_artifact = (
            await session.execute(
                select(RunArtifact).where(
                    RunArtifact.run_id == run_id,
                    RunArtifact.artifact_key == SITE_PAGE_MANIFEST_KEY,
                    RunArtifact.status == "completed",
                    RunArtifact.prompt_version == SITE_PAGE_MANIFEST_VERSION,
                )
            )
        ).scalar_one_or_none()

    manifest = _validated_site_page_manifest(
        manifest_artifact,
        run_domain=str(run.domain or ""),
    )
    manifest_urls = _manifest_page_urls(manifest)
    if manifest_urls:
        pages = [page for page in pages if page.url in manifest_urls]

    latest: dict[tuple[str, str, ProbeType], DomainProbe] = {}
    for probe in probes:
        latest[(probe.target_url, probe.user_agent_label, probe.probe_type)] = probe
    page_groups: dict[str, dict[str, DomainProbe]] = defaultdict(dict)
    for probe in latest.values():
        if probe.probe_type is ProbeType.main_page:
            page_groups[probe.target_url][probe.user_agent_label] = probe

    known_checks = 0
    passed_checks = 0
    unknown_checks = 0
    family_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    family_unknown_counts: dict[str, int] = defaultdict(int)
    page_results: list[dict[str, Any]] = []
    conditional_blocks = 0
    auth_forms = 0
    auth_walls = 0
    utility_pages = 0
    for page in pages:
        is_utility = _is_utility_page(page.url, page.page_kind)
        utility_pages += int(is_utility)
        variants = page_groups.get(page.url, {})
        control = variants.get(CONTROL_LABEL)
        baseline = int(control.content_extractable_text_length or 0) if control else None
        accessible_families: set[str] = set()
        blocked_families: set[str] = set()
        unknown_families: set[str] = set()
        page_known_checks = 0
        page_passed_checks = 0
        page_unknown_checks = 0
        for label, probe in variants.items():
            if label not in AI_LABELS:
                continue
            family = _family_for_label(label)
            outcome = _probe_access_outcome(probe, baseline)
            if outcome == "unknown":
                page_unknown_checks += 1
                unknown_families.add(family)
            elif outcome == "available":
                page_known_checks += 1
                page_passed_checks += 1
                accessible_families.add(family)
            else:
                page_known_checks += 1
                blocked_families.add(family)
            signals = probe.content_signals or {}
            if not is_utility:
                if outcome == "unknown":
                    family_unknown_counts[family] += 1
                else:
                    family_counts[family][1] += 1
                    if outcome == "available":
                        family_counts[family][0] += 1
                auth_forms += int(bool(signals.get("auth_form_present")))
                auth_walls += int(bool(signals.get("looks_like_login_wall")))
                conditional_blocks += int(bool(signals.get("ua_conditional_block")))
        if not is_utility:
            known_checks += page_known_checks
            passed_checks += page_passed_checks
            unknown_checks += page_unknown_checks
        page_signals = page.content_signals or {}
        body_truncated = bool(page_signals.get("_body_truncated"))
        structured_data_complete = (
            not body_truncated
            and page_signals.get("structured_data_complete") is not False
        )
        structured_data_types = page_signals.get("structured_data_types") or []
        page_results.append(
            {
                "url": page.url,
                "page_kind": page.page_kind,
                "title": page.title or page.url,
                "http_status": page.http_status,
                "text_length": page.text_length,
                "render_strategy": (
                    "unknown"
                    if body_truncated
                    else page_signals.get("render_strategy", "unknown")
                ),
                "render_confidence": page_signals.get(
                    "render_strategy_confidence", "unknown"
                ),
                "body_truncated": body_truncated,
                "structured_data_complete": structured_data_complete,
                "structured_data_types": structured_data_types,
                "entity_structured_data_types": _entity_structured_data_types(
                    structured_data_types
                ),
                "accessible_families": sorted(accessible_families),
                "blocked_families": sorted(blocked_families - accessible_families),
                "unknown_families": sorted(
                    unknown_families - accessible_families - blocked_families
                ),
                "passed_checks": page_passed_checks,
                "total_checks": page_known_checks,
                "unknown_checks": page_unknown_checks,
                "expected_checks": page_known_checks + page_unknown_checks,
                "access_rate": _rate(page_passed_checks, page_known_checks),
                "is_utility": is_utility,
            }
        )

    robots_latest: dict[str, RobotsRule] = {}
    for row in robots:
        robots_latest[row.bot_name] = row
    relevant_robot_names = (
        "GPTBot",
        "OAI-SearchBot",
        "ChatGPT-User",
        "ClaudeBot",
        "PerplexityBot",
        "Perplexity-User",
        "Google-Extended",
        "Googlebot",
        "DeepSeekBot",
    )
    content_urls = [
        page.url
        for page in pages
        if not _is_utility_page(page.url, page.page_kind)
    ]
    robot_allowed = 0
    robot_known = 0
    robot_unknown = 0
    partial_robot_names: set[str] = set()
    fully_disallowed_names: set[str] = set()
    for name in relevant_robot_names:
        row = robots_latest.get(name)
        if row is None:
            robot_unknown += max(1, len(content_urls))
            continue
        if row.rule == "partial":
            partial_robot_names.add(name)
        outcomes: list[bool | None] = []
        for url in content_urls or [f"https://{run.domain or ''}/"]:
            if row.rule in {"allow_all", "not_mentioned"}:
                outcome: bool | None = True
            elif row.rule == "disallow_all":
                outcome = False
            else:
                outcome = robots_path_allowed(row.raw_directives, url)
            outcomes.append(outcome)
            if outcome is None:
                robot_unknown += 1
            else:
                robot_known += 1
                robot_allowed += int(outcome)
        known_outcomes = [outcome for outcome in outcomes if outcome is not None]
        if known_outcomes and not any(known_outcomes):
            fully_disallowed_names.add(name)
    fetch_ratio = passed_checks / known_checks if known_checks else None
    robots_ratio = robot_allowed / robot_known if robot_known else None
    robots_state = (
        "unknown"
        if not robot_known
        else "partial"
        if partial_robot_names
        or robot_unknown
        or (robots_ratio is not None and 0 < robots_ratio < 1)
        else "closed"
        if robots_ratio == 0
        else "open"
    )

    render_counts: Counter[str] = Counter()
    for page in pages:
        if _is_utility_page(page.url, page.page_kind):
            continue
        signals = page.content_signals or {}
        render_counts[
            "unknown"
            if signals.get("_body_truncated")
            else str(signals.get("render_strategy") or "unknown")
        ] += 1
    rendering = _rendering_assessment(render_counts)
    render_ratio = rendering["ratio"]
    content_page_count = max(0, len(pages) - utility_pages)
    structured_known_pages = [
        page
        for page in pages
        if not _is_utility_page(page.url, page.page_kind)
        and not bool((page.content_signals or {}).get("_body_truncated"))
        and (page.content_signals or {}).get("structured_data_complete") is not False
    ]
    structured_pages = sum(
        bool(
            _entity_structured_data_types(
                (page.content_signals or {}).get("structured_data_types")
            )
        )
        for page in structured_known_pages
    )
    structured_total = len(structured_known_pages)
    structured_unknown_pages = max(0, content_page_count - structured_total)
    page_coverage = _technical_page_coverage(content_page_count, manifest)
    structured_ratio = (
        structured_pages / structured_total if structured_total else None
    )

    weighted_parts: list[tuple[float, float]] = []
    if fetch_ratio is not None:
        weighted_parts.append((fetch_ratio, 0.60))
    if robots_ratio is not None:
        weighted_parts.append((robots_ratio, 0.20))
    if render_ratio is not None:
        weighted_parts.append((render_ratio, 0.15))
    if structured_ratio is not None:
        weighted_parts.append((structured_ratio, 0.05))
    access_score = (
        round(
            sum(value * weight for value, weight in weighted_parts)
            / sum(weight for _, weight in weighted_parts)
            * 100
        )
        if weighted_parts
        else None
    )

    barriers: list[dict[str, str]] = []
    if conditional_blocks:
        barriers.append(
            {
                "severity": "critical",
                "title": "Часть ИИ-краулеров получает другую версию сайта",
                "evidence": "Обычный браузер читает страницу, а отдельные ИИ-краулеры получают блокировку или урезанный ответ.",
                "action": "Сверить правила WAF и CDN для поисковых и пользовательских ИИ-краулеров.",
            }
        )
    if render_counts["client_rendered_shell"]:
        barriers.append(
            {
                "severity": "critical",
                "title": "Часть содержания появляется только после запуска JavaScript",
                "evidence": "В исходном HTML ключевых страниц мало читаемого текста, основное содержание добавляет браузер.",
                "action": "Отдавать основной текст и метаданные с сервера или через статическую генерацию.",
            }
        )
    disallowed = sorted(fully_disallowed_names)
    if disallowed:
        barriers.append(
            {
                "severity": "important",
                "title": "Правила обхода закрывают отдельные ИИ-системы",
                "evidence": "В robots.txt есть полный запрет для части релевантных краулеров.",
                "action": "Проверить, какие запреты соответствуют политике бренда, а какие остались случайно.",
            }
        )
    if auth_walls:
        barriers.append(
            {
                "severity": "important",
                "title": "Отдельные страницы требуют входа до чтения содержания",
                "evidence": "На странице форма входа доминирует, а открытого основного текста недостаточно.",
                "action": "Оставить индексируемое описание предложения до формы входа.",
            }
        )
    if not structured_pages and structured_total:
        barriers.append(
            {
                "severity": "observation",
                "title": "Страницы почти не объясняют свои сущности через разметку",
                "evidence": (
                    f"На {structured_total} полностью прочитанных страницах не найдена "
                    "структурированная разметка с типами организации, продукта или статьи."
                ),
                "action": "Добавить корректную Schema.org-разметку там, где она соответствует видимому содержанию.",
            }
        )

    family_rows = []
    for family in ("OpenAI", "Anthropic", "Perplexity", "Google", "DeepSeek"):
        passed, total = family_counts.get(family, [0, 0])
        unknown = family_unknown_counts.get(family, 0)
        family_rows.append(
            {
                "name": family,
                "access_rate": round(passed / total * 100) if total else None,
                "passed_count": passed,
                "total_count": total,
                "unknown_count": unknown,
                "expected_count": total + unknown,
                "state": (
                    "available"
                    if total and passed == total and not unknown
                    else "blocked"
                    if total and not passed and not unknown
                    else "partial"
                    if total
                    else "unknown"
                ),
            }
        )

    return {
        "score": access_score,
        "evaluated_pages": page_coverage["evaluated_pages"],
        "discovered_pages": page_coverage["discovered_pages"],
        "coverage_rate": page_coverage["coverage_rate"],
        "coverage_state": page_coverage["coverage_state"],
        "coverage": page_coverage,
        "state": (
            "available"
            if access_score is not None and access_score >= 85
            else "partial"
            if access_score is not None and access_score >= 50
            else "blocked"
            if access_score is not None
            else "unknown"
        ),
        "families": family_rows,
        "pages": page_results,
        "summary": {
            "content_pages_evaluated": content_page_count,
            "evaluated_pages": page_coverage["evaluated_pages"],
            "discovered_pages": page_coverage["discovered_pages"],
            "coverage_rate": page_coverage["coverage_rate"],
            "coverage_state": page_coverage["coverage_state"],
            "utility_pages_excluded": utility_pages,
            "passed_checks": passed_checks,
            "total_checks": known_checks,
            "unknown_checks": unknown_checks,
            "expected_checks": known_checks + unknown_checks,
            "structured_pages": structured_pages,
            "structured_total": structured_total,
            "structured_unknown_pages": structured_unknown_pages,
            "facts": [
                {
                    "key": "robots",
                    "label": "Правила robots.txt",
                    "value": (
                        "Открыты"
                        if robots_state == "open"
                        else "Открыты частично"
                        if robots_state == "partial"
                        else "Закрыты"
                        if robots_state == "closed"
                        else "Не удалось проверить"
                    ),
                    "state": (
                        "good"
                        if robots_state == "open"
                        else "warning"
                        if robots_state == "partial"
                        else "bad"
                        if robots_state == "closed"
                        else "unknown"
                    ),
                },
                {
                    "key": "server_html",
                    "label": "Основной текст с сервера",
                    "value": (
                        (
                            f"{round(rendering['evaluated_pages'] * render_ratio)} из "
                            f"{content_page_count} подтверждены"
                            + (
                                f"; {rendering['unknown_pages']} "
                                "не определено"
                                if rendering["unknown_pages"]
                                else ""
                            )
                        )
                        if content_page_count and render_ratio is not None
                        else (
                            f"{rendering['unknown_pages']} из "
                            f"{content_page_count} не определено"
                        )
                        if content_page_count
                        else "Не удалось определить"
                    ),
                    "state": (
                        "good"
                        if (
                            render_ratio is not None
                            and render_ratio >= 0.85
                            and not rendering["unknown_pages"]
                        )
                        else "warning"
                        if render_ratio is not None and render_ratio > 0
                        else "bad"
                        if render_ratio == 0
                        else "unknown"
                    ),
                },
                {
                    "key": "auth",
                    "label": "Открытое содержание",
                    "value": (
                        "Часть страниц закрыта входом"
                        if auth_walls
                        else "Доступно без входа"
                    ),
                    "state": "bad" if auth_walls else "good",
                },
                {
                    "key": "structured_data",
                    "label": "Машинное описание сущностей",
                    "value": (
                        (
                            f"{structured_pages} из {structured_total}"
                            + (
                                f"; {structured_unknown_pages} не определено"
                                if structured_unknown_pages
                                else ""
                            )
                        )
                        if structured_total
                        else "Не удалось проверить"
                    ),
                    "state": (
                        "good"
                        if structured_ratio is not None and structured_ratio >= 0.75
                        else "warning"
                        if structured_ratio is not None and structured_ratio > 0
                        else "bad"
                        if structured_ratio == 0
                        else "unknown"
                    ),
                },
            ],
        },
        "barriers": barriers,
        "robots": {
            "state": robots_state,
            "allowed_checks": robot_allowed,
            "known_checks": robot_known,
            "unknown_checks": robot_unknown,
            "partial_rules": sorted(partial_robot_names),
            "disallowed_families": sorted(
                {_family_for_label(name) for name in disallowed}
            ),
        },
        "rendering": {
            "counts": dict(render_counts),
            "server_readable_share": round(render_ratio * 100) if render_ratio is not None else None,
            "evaluated_pages": rendering["evaluated_pages"],
            "unknown_pages": rendering["unknown_pages"],
            "conclusion": rendering["conclusion"],
        },
        "structured_data": {
            "entity_pages": structured_pages,
            "evaluated_pages": structured_total,
            "unknown_pages": structured_unknown_pages,
            "entity_share": (
                round(structured_ratio * 100)
                if structured_ratio is not None
                else None
            ),
        },
        "auth": {
            "forms_present": auth_forms > 0,
            "walls_detected": auth_walls > 0,
            "interpretation": (
                "Формы входа встречаются, но содержательные страницы остаются читаемыми."
                if auth_forms and not auth_walls
                else "Формы входа не мешают чтению открытого содержания."
                if not auth_walls
                else "На отдельных страницах вход закрывает основное содержание."
            ),
        },
    }


async def _classify_site(run_id: str, site_context: dict[str, Any]) -> dict[str, Any]:
    system = f"""
Ты классифицируешь один сайт только по переданному серверному содержанию.
Не используй веб-поиск и внешние знания. Определи бренд, тип сайта, категорию,
тематики, рынок, бизнес-модель, продукты, аудитории и географию. Выдели
задачи, с которыми клиенты приходят на этот рынок, и критерии выбора решения.
Построй entity_scope: отдели основной бренд от подтверждённых бизнес-направлений,
дочерних брендов, продуктов, сервисов и платформ, которые представляет сайт.
Для каждой сущности укажи связь с основным брендом и коммерческую значимость.
Не считай обычный пункт меню или стороннего партнёра частью целевого портфеля.
Поле requested_site — это домен, который ввёл пользователь; анализируй именно
его и подтверждай выводы содержимым pages. Не додумывай: сомнения запиши отдельно.
Название бренда подтверждай заголовком, описанием или повторяющимся текстом.

{LIVE_RUSSIAN_RULES}
""".strip()
    return await _structured_artifact(
        run_id,
        stage_key="scenario_design",
        artifact_key="site_profile",
        schema=SITE_PROFILE_SCHEMA,
        schema_name="aiv_site_profile",
        system=system,
        user_payload=site_context,
        model=ANALYSIS_MODEL,
        reasoning_effort="high",
    )


# Структурный дефект самого ревью (не сценариев): не тот состав checks.
# Такой вердикт не должен тратить попытку генерации набора — перегенерировать
# нужно ревью, а не сценарии.
_REVIEW_STRUCTURALLY_INVALID = (
    "Критик должен вернуть ровно по одной проверке на каждый из шести "
    "безбрендовых сценариев."
)


class MarketResearchGateError(OpenRouterError):
    """Market context is not sufficiently evidenced to design scenarios."""


_MARKET_RESEARCH_DIMENSIONS = (
    "market",
    "topics",
    "geography",
    "audiences",
    "customer_jobs",
    "decision_criteria",
    "terminology",
)


def _nonempty_research_items(value: Any) -> list[Any]:
    if not isinstance(value, list):
        return []
    return [
        item
        for item in value
        if (
            isinstance(item, dict)
            and any(str(part or "").strip() for part in item.values())
        )
        or (not isinstance(item, dict) and str(item or "").strip())
    ]


def _normalized_research_url(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        parsed = urlparse(raw)
    except ValueError:
        return ""
    scheme = parsed.scheme.casefold()
    host = (parsed.hostname or "").casefold().removeprefix("www.")
    if scheme not in {"http", "https"} or not host:
        return ""
    port = f":{parsed.port}" if parsed.port else ""
    path = re.sub(r"/+", "/", parsed.path or "/")
    if path != "/":
        path = path.rstrip("/")
    query = f"?{parsed.query}" if parsed.query else ""
    return f"{scheme}://{host}{port}{path}{query}"


_EVIDENCE_URL_RX = re.compile(r"https?://[^\s\"'<>\)\]]+")


def _urls_in_evidence_text(value: Any) -> set[str]:
    """Extract confirmed-citation candidates from a prose evidence line.

    Критик пишет supporting_evidence прозой с URL внутри текста — именно так,
    как просит промпт («подтверждения и URL»). Парсить строку целиком как URL
    нельзя: предложение не URL, множество получается пустым, и прошедший
    ревью набор детерминированно отвергается с бессмысленным «Исправление не
    требуется» (прогон 5ae13350, 2026-08-21).
    """

    text = str(value or "")
    found: set[str] = set()
    for match in _EVIDENCE_URL_RX.findall(text):
        normalized = _normalized_research_url(match.rstrip(".,;:!?"))
        if normalized:
            found.add(normalized)
    return found


def _research_url_host(value: Any) -> str:
    normalized = _normalized_research_url(value)
    if not normalized:
        return ""
    return (urlparse(normalized).hostname or "").casefold().removeprefix("www.")


def _same_site_host(host: str, requested_domain: str) -> bool:
    host = str(host or "").casefold().removeprefix("www.").strip(".")
    requested_domain = (
        str(requested_domain or "")
        .casefold()
        .removeprefix("www.")
        .strip(".")
    )
    if not host or not requested_domain:
        return False
    return (
        host == requested_domain
        or host.endswith(f".{requested_domain}")
        or requested_domain.endswith(f".{host}")
    )


def _market_research_input(
    profile: dict[str, Any],
    site_context: dict[str, Any],
) -> dict[str, Any]:
    return {
        "requested_site": copy.deepcopy(site_context.get("requested_site") or {}),
        "site_profile": copy.deepcopy(profile),
        "site_evidence": copy.deepcopy(site_context.get("pages") or []),
    }


def _site_profile_research_errors(
    profile: dict[str, Any],
    payload: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    requested_site = payload.get("requested_site")
    if not isinstance(requested_site, dict) or not str(
        requested_site.get("domain") or ""
    ).strip():
        errors.append("Не подтверждён домен исследуемого сайта.")
    if not str(profile.get("brand_name") or "").strip():
        errors.append("Сайт не подтвердил название основного бренда.")
    if not _nonempty_research_items(profile.get("evidence")):
        errors.append("В профиле нет доказательств, извлечённых с сайта.")
    pages = payload.get("site_evidence")
    if not isinstance(pages, list) or not any(
        isinstance(page, dict)
        and str(page.get("url") or "").strip()
        and any(
            str(page.get(key) or "").strip()
            for key in ("title", "meta_description", "main_text")
        )
        for page in pages
    ):
        errors.append("Сайт не дал содержательных доказательств для идентификации.")
    return errors


def _market_research_sufficiency(
    research: dict[str, Any],
    profile: dict[str, Any],
    *,
    requested_site: dict[str, Any] | None,
    site_evidence: list[dict[str, Any]] | None,
    web_attestation: dict[str, Any] | None,
    citation_urls: list[str] | None,
) -> dict[str, Any]:
    """Fail closed when market context is incomplete or not web-attested."""

    blocking: list[str] = []
    limited: list[str] = []
    requested_site = requested_site or {}
    requested_domain = str(requested_site.get("domain") or "").strip()
    expected_brand = str(profile.get("brand_name") or "").strip()
    site_confirmed = research.get("site_confirmed")
    external = research.get("external_market_research")
    sources = research.get("sources")
    site_confirmed = site_confirmed if isinstance(site_confirmed, dict) else {}
    external = external if isinstance(external, dict) else {}
    sources = sources if isinstance(sources, list) else []

    actual_brand = str(site_confirmed.get("primary_brand") or "").strip()
    if not expected_brand or actual_brand.casefold() != expected_brand.casefold():
        blocking.append(
            "Внешнее исследование изменило или потеряло основной бренд, "
            "подтверждённый сайтом."
        )

    attestation = web_attestation if isinstance(web_attestation, dict) else {}
    if (
        attestation.get("version") != WEB_ATTESTATION_VERSION
        or attestation.get("policy") != WebSearchPolicy.REQUIRED.value
        or attestation.get("state") != "verified"
        or attestation.get("metric_eligible") is not True
        or bool(attestation.get("violations"))
        or not isinstance(attestation.get("web_search_requests"), int)
        or int(attestation.get("web_search_requests") or 0) < 1
    ):
        blocking.append("Веб-поиск не прошёл обязательную аттестацию.")

    confirmed_citations = {
        normalized
        for value in citation_urls or []
        if (normalized := _normalized_research_url(value))
    }
    if not confirmed_citations:
        blocking.append("OpenRouter не подтвердил ни одной URL-цитаты.")

    declared_source_urls: set[str] = set()
    confirmed_source_urls: set[str] = set()
    external_confirmed_source_urls: set[str] = set()
    for source in sources:
        if not isinstance(source, dict):
            continue
        normalized = _normalized_research_url(source.get("url"))
        if not normalized:
            continue
        declared_source_urls.add(normalized)
        if normalized not in confirmed_citations:
            continue
        confirmed_source_urls.add(normalized)
        if not _same_site_host(_research_url_host(normalized), requested_domain):
            external_confirmed_source_urls.add(normalized)
        if str(source.get("confidence") or "") == "low":
            limited.append(
                f"Источник {normalized} помечен низкой уверенностью."
            )
    unconfirmed_declared_sources = declared_source_urls - confirmed_source_urls
    for normalized in sorted(unconfirmed_declared_sources):
        limited.append(
            f"Источник {normalized} не подтверждён URL-цитатой OpenRouter."
        )
    if len(external_confirmed_source_urls) < 2:
        limited.append(
            "Нужно не менее двух подтверждённых внешних источников о рынке."
        )

    site_evidence_urls = {
        normalized
        for page in site_evidence or []
        if isinstance(page, dict)
        and (normalized := _normalized_research_url(page.get("url")))
    }
    confirmed_site_claims = 0
    for item in _nonempty_research_items(site_confirmed.get("evidence")):
        if not isinstance(item, dict):
            continue
        normalized = _normalized_research_url(item.get("url"))
        if (
            normalized
            and normalized in site_evidence_urls
            and str(item.get("claim") or "").strip()
            and str(item.get("excerpt") or "").strip()
        ):
            confirmed_site_claims += 1
    if confirmed_site_claims < 1:
        blocking.append(
            "В site-confirmed facts нет доказательства с проверенной страницы сайта."
        )

    missing_dimensions: list[str] = []
    for dimension in _MARKET_RESEARCH_DIMENSIONS:
        value = external.get(dimension)
        if dimension == "market":
            present = bool(str(value or "").strip())
        else:
            present = bool(_nonempty_research_items(value))
        if not present:
            missing_dimensions.append(dimension)
    if "customer_jobs" in missing_dimensions:
        blocking.append("Не подтверждены реальные задачи аудитории.")
    if "decision_criteria" in missing_dimensions:
        blocking.append("Не подтверждены критерии выбора.")
    for dimension in missing_dimensions:
        if dimension not in {"customer_jobs", "decision_criteria"}:
            limited.append(f"Не заполнено обязательное измерение {dimension}.")

    evidence_dimensions: set[str] = set()
    weak_evidence_dimensions: set[str] = set()
    for item in _nonempty_research_items(external.get("evidence")):
        if not isinstance(item, dict):
            continue
        dimension = str(item.get("dimension") or "")
        source_urls = {
            normalized
            for value in item.get("source_urls") or []
            if (normalized := _normalized_research_url(value))
        }
        for normalized in sorted(source_urls - external_confirmed_source_urls):
            limited.append(
                f"Доказательство {dimension or 'без измерения'} ссылается "
                f"на неподтверждённый внешний источник {normalized}."
            )
        if (
            dimension in _MARKET_RESEARCH_DIMENSIONS
            and str(item.get("claim") or "").strip()
            and str(item.get("evidence") or "").strip()
            and bool(source_urls & external_confirmed_source_urls)
        ):
            evidence_dimensions.add(dimension)
            if str(item.get("confidence") or "") == "low":
                weak_evidence_dimensions.add(dimension)

    for dimension in _MARKET_RESEARCH_DIMENSIONS:
        if dimension not in evidence_dimensions:
            message = (
                f"Измерение {dimension} не связано с подтверждённой "
                "внешней URL-цитатой."
            )
            if dimension in {"customer_jobs", "decision_criteria"}:
                blocking.append(message)
            else:
                limited.append(message)
    for dimension in sorted(weak_evidence_dimensions):
        limited.append(
            f"Измерение {dimension} опирается только на доказательство "
            "низкой уверенности."
        )

    if str(research.get("confidence") or "") == "low":
        limited.append("Общая уверенность исследования отмечена как низкая.")

    blocking = list(dict.fromkeys(blocking))
    limited = list(dict.fromkeys(limited))
    status = "blocked" if blocking else "limited" if limited else "ready"
    return {
        "status": status,
        "blocking_issues": blocking,
        "limited_issues": limited,
        "required_dimensions": list(_MARKET_RESEARCH_DIMENSIONS),
        "evidenced_dimensions": sorted(evidence_dimensions),
        "confirmed_source_urls": sorted(confirmed_source_urls),
        "confirmed_external_source_urls": sorted(
            external_confirmed_source_urls
        ),
        "confirmed_site_claims": confirmed_site_claims,
    }


def _market_research_with_gate(
    research: dict[str, Any],
    profile: dict[str, Any],
    *,
    payload: dict[str, Any],
    web_attestation: dict[str, Any] | None,
    citations: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    citation_items = [
        item
        for item in citations or []
        if isinstance(item, dict) and str(item.get("url") or "").strip()
    ]
    output = copy.deepcopy(research)
    gate = _market_research_sufficiency(
        output,
        profile,
        requested_site=payload.get("requested_site"),
        site_evidence=payload.get("site_evidence"),
        web_attestation=web_attestation,
        citation_urls=[str(item["url"]) for item in citation_items],
    )
    output["status"] = gate["status"]
    output["sufficiency"] = gate
    output["web_evidence"] = {
        "attestation": copy.deepcopy(web_attestation or {}),
        "citations": copy.deepcopy(citation_items),
    }
    return output


def _require_market_research_usable(
    research: dict[str, Any] | None,
) -> dict[str, Any]:
    """Fail on blocked or unattested research; let limited proceed.

    limited — штатный честный итог: схема и промпт сами предписывают модели
    ставить его при неопределённостях, а весь дальнейший пайплайн (метрики,
    coverage, отчёт) умеет деградировать с data_state="limited". Требовать
    здесь идеального ready — значит отвергать состояние, которое мы же
    запросили; фатальны только blocked и провал веб-аттестации.
    """

    if not isinstance(research, dict):
        raise MarketResearchGateError(
            "Нельзя строить сценарии без исследования рынка."
        )
    sufficiency = research.get("sufficiency")
    web_evidence = research.get("web_evidence")
    attestation = (
        web_evidence.get("attestation")
        if isinstance(web_evidence, dict)
        else None
    )
    if (
        research.get("status") not in {"ready", "limited"}
        or not isinstance(sufficiency, dict)
        or sufficiency.get("status") not in {"ready", "limited"}
        or sufficiency.get("blocking_issues")
        or not isinstance(attestation, dict)
        or attestation.get("version") != WEB_ATTESTATION_VERSION
        or attestation.get("policy") != WebSearchPolicy.REQUIRED.value
        or attestation.get("state") != "verified"
        or attestation.get("metric_eligible") is not True
    ):
        issues: list[str] = []
        if isinstance(sufficiency, dict):
            issues.extend(sufficiency.get("blocking_issues") or [])
            issues.extend(sufficiency.get("limited_issues") or [])
        raise MarketResearchGateError(
            "; ".join(str(issue) for issue in issues if str(issue).strip())
            or "Исследование рынка не прошло проверку достаточности."
        )
    return research


async def _cached_market_research(
    run_id: str,
    *,
    profile: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    async with SessionLocal() as session:
        artifact = (
            await session.execute(
                select(RunArtifact).where(
                    RunArtifact.run_id == run_id,
                    RunArtifact.artifact_key == "market_research",
                )
            )
        ).scalar_one_or_none()
    if artifact is None or not _artifact_cache_matches(
        artifact,
        input_json=payload,
        model=ANALYSIS_MODEL,
        prompt_version=MARKET_RESEARCH_VERSION,
    ):
        return None
    if not isinstance(artifact.output_json, dict):
        return None
    usage = artifact.usage_json if isinstance(artifact.usage_json, dict) else {}
    attestation = usage.get("_aiv_web_attestation")
    annotations = usage.get("_aiv_response_annotations")
    citations = []
    for annotation in annotations if isinstance(annotations, list) else []:
        if not isinstance(annotation, dict):
            continue
        source = annotation.get("url_citation")
        if not isinstance(source, dict):
            continue
        url = str(source.get("url") or "").strip()
        if url:
            citations.append(
                {
                    "url": url,
                    "title": str(source.get("title") or ""),
                    "content": str(source.get("content") or ""),
                }
            )
    return _market_research_with_gate(
        artifact.output_json,
        profile,
        payload=payload,
        web_attestation=attestation if isinstance(attestation, dict) else None,
        citations=citations,
    )


async def _market_research(
    run_id: str,
    profile: dict[str, Any],
    site_context: dict[str, Any],
) -> dict[str, Any]:
    """Research the market with attested web search before prompt design."""

    payload = _market_research_input(profile, site_context)
    profile_errors = _site_profile_research_errors(profile, payload)
    if profile_errors:
        await _save_artifact(
            run_id,
            stage_key="scenario_design",
            artifact_key="market_research",
            status="failed",
            model=ANALYSIS_MODEL,
            input_json=payload,
            output_json={
                "status": "blocked",
                "sufficiency": {
                    "status": "blocked",
                    "blocking_issues": profile_errors,
                    "limited_issues": [],
                },
            },
            error_message="; ".join(profile_errors),
            prompt_version=MARKET_RESEARCH_VERSION,
        )
        raise MarketResearchGateError("; ".join(profile_errors))

    cached = await _cached_market_research(
        run_id,
        profile=profile,
        payload=payload,
    )
    if isinstance(cached, dict):
        return _require_market_research_usable(cached)

    await _save_artifact(
        run_id,
        stage_key="scenario_design",
        artifact_key="market_research",
        status="running",
        model=ANALYSIS_MODEL,
        input_json=payload,
        prompt_version=MARKET_RESEARCH_VERSION,
    )
    # Шаг разделён на два вызова, потому что строгая JSON-схема и
    # URL-цитаты в агентском цикле веб-поиска взаимоисключаются (замер
    # 2026-08-21 на anthropic/claude-opus-5): эндпоинты, соблюдающие
    # response_format, не отдают url_citation-аннотаций, а эндпоинты и
    # engine=exa, отдающие цитаты, игнорируют схему и пишут повествование.
    # Поэтому retrieval и структурирование аттестуются раздельно:
    # 1) исследование со свободным текстом и обязательным веб-поиском —
    #    отсюда берутся аттестация и подтверждённые цитаты;
    # 2) структурирование в MARKET_RESEARCH_SCHEMA без доступа к вебу.
    research_system = f"""
Ты старший исследователь рынка. Перед проектированием поисковых сценариев
изучи сайт, который указал пользователь, и внешний рыночный контекст.
Обязательно используй предоставленный веб-поиск и опирай каждый внешний
вывод на URL-источник из результатов поиска.

Разделяй два слоя:
1. site_confirmed — только факты, которые уже подтверждены site_profile и
страницами в site_evidence;
2. внешние рыночные сведения — рынок, тематика, география, аудитории,
реальные customer jobs, критерии выбора и профессиональная терминология
(семь измерений, каждое с точной мыслью, выдержкой и URL источника).

Основной бренд определяет только сайт. Никогда не заменяй primary_brand
названием группы, конкурента, отрасли или бренда из внешнего поиска.

Пиши структурированные исследовательские заметки: для каждого измерения —
отдельный раздел с выводами, дословными выдержками и полными URL. Используй
не менее двух независимых внешних источников. Не заполняй пробелы догадками:
неизвестное перечисли отдельным разделом «Неопределённости».

{LIVE_RUSSIAN_RULES}
""".strip()
    try:
        research = await chat(
            model=ANALYSIS_MODEL,
            messages=[
                {"role": "system", "content": research_system},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False),
                },
            ],
            web_policy=WebSearchPolicy.REQUIRED,
            reasoning_effort="high",
            max_tokens=12_000,
            temperature=0.1,
        )
    except OpenRouterPolicyError as exc:
        output = _market_research_with_gate(
            {},
            profile,
            payload=payload,
            web_attestation=exc.result.web_attestation,
            citations=exc.result.citations,
        )
        await _save_artifact(
            run_id,
            stage_key="scenario_design",
            artifact_key="market_research",
            status="failed",
            model=ANALYSIS_MODEL,
            input_json=payload,
            output_json=output,
            raw_text=exc.result.text,
            usage_json=exc.result.usage,
            error_message=str(exc),
            prompt_version=MARKET_RESEARCH_VERSION,
        )
        raise MarketResearchGateError(
            "Исследование рынка не подтвердило обязательный веб-поиск."
        ) from exc
    except Exception as exc:
        await _save_artifact(
            run_id,
            stage_key="scenario_design",
            artifact_key="market_research",
            status="failed",
            model=ANALYSIS_MODEL,
            input_json=payload,
            error_message=str(exc),
            prompt_version=MARKET_RESEARCH_VERSION,
        )
        raise

    confirmed_urls = sorted(
        {
            str(item.get("url") or "").strip()
            for item in research.citations or []
            if isinstance(item, dict) and str(item.get("url") or "").strip()
        }
    )
    structuring_system = f"""
Ты редактор данных. Преобразуй исследовательские заметки в JSON строго по
схеме, ничего не добавляя от себя.

Правила:
1. site_confirmed заполняй только фактами, подтверждёнными site_profile и
страницами site_evidence из входных данных; в evidence указывай URL страницы
сайта, точный claim и дословный excerpt.
2. external_market_research заполняй только сведениями из заметок. В
source_urls каждого доказательства используй только URL из списка
confirmed_source_urls; источник, которого нет в списке, перечисли в
uncertainties, а не в source_urls.
3. Никогда не меняй primary_brand: его определяет сайт.
4. Не заполняй пробелы догадками: неизвестность запиши в uncertainties и
поставь limited или blocked. ready допустим только при полном покрытии
измерений подтверждёнными источниками.

{LIVE_RUSSIAN_RULES}
""".strip()
    try:
        structured = await chat(
            model=ANALYSIS_MODEL,
            messages=[
                {"role": "system", "content": structuring_system},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "site_input": payload,
                            "research_notes": research.text,
                            "confirmed_source_urls": confirmed_urls,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            response_schema=MARKET_RESEARCH_SCHEMA,
            schema_name="aiv_market_research",
            web_policy=WebSearchPolicy.FORBIDDEN,
            reasoning_effort="high",
            max_tokens=12_000,
            temperature=0.1,
        )
    except Exception as exc:
        await _save_artifact(
            run_id,
            stage_key="scenario_design",
            artifact_key="market_research",
            status="failed",
            model=ANALYSIS_MODEL,
            input_json=payload,
            raw_text=research.text,
            usage_json=research.usage,
            error_message=str(exc),
            prompt_version=MARKET_RESEARCH_VERSION,
        )
        raise

    combined_usage = dict(research.usage)
    combined_usage["_aiv_structuring_usage"] = structured.usage
    combined_raw = (
        f"{research.text}\n\n=== STRUCTURED JSON ===\n\n{structured.text}"
    )
    if not isinstance(structured.parsed, dict):
        error = "Market research response is not an object"
        await _save_artifact(
            run_id,
            stage_key="scenario_design",
            artifact_key="market_research",
            status="failed",
            model=ANALYSIS_MODEL,
            input_json=payload,
            raw_text=combined_raw,
            usage_json=combined_usage,
            error_message=error,
            prompt_version=MARKET_RESEARCH_VERSION,
        )
        raise OpenRouterError(error)
    # Гейт и аттестация — от исследовательского вызова: именно он ходил в веб.
    output = _market_research_with_gate(
        structured.parsed,
        profile,
        payload=payload,
        web_attestation=research.web_attestation,
        citations=research.citations,
    )
    await _save_artifact(
        run_id,
        stage_key="scenario_design",
        artifact_key="market_research",
        status="completed",
        model=ANALYSIS_MODEL,
        input_json=payload,
        output_json=output,
        raw_text=combined_raw,
        usage_json=combined_usage,
        prompt_version=MARKET_RESEARCH_VERSION,
    )
    return _require_market_research_usable(output)


def _prompt_contains_alias(text: str, alias: str) -> bool:
    """Match a brand phrase without confusing it with part of another word."""

    normalized_text = str(text or "").casefold().replace("ё", "е")
    normalized_alias = str(alias or "").casefold().replace("ё", "е").strip()
    if len(normalized_alias) < 2:
        return False
    return bool(
        re.search(
            rf"(?<![\w]){re.escape(normalized_alias)}(?![\w])",
            normalized_text,
            re.UNICODE,
        )
    )


def _validate_prompt_set(
    prompt_set: dict[str, Any],
    profile: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    prompts = prompt_set.get("prompts")
    if not isinstance(prompts, list) or len(prompts) != 9:
        return ["Нужно ровно девять сценариев."]
    unbranded = [item for item in prompts if item.get("role") == "unbranded_discovery"]
    branded = [item for item in prompts if item.get("role") == "brand_diagnostic"]
    if len(unbranded) != 6 or len(branded) != 3:
        errors.append("Нужно шесть безбрендовых и три брендовых сценария.")
    expected = {"I", "E", "T", "NB", "NAV", "TR"}
    if {item.get("intent_class") for item in unbranded} != expected:
        errors.append("Безбрендовый набор должен покрывать I, E, T, NB, NAV и TR по одному разу.")
    aliases = [
        str(value).strip()
        for value in [profile.get("brand_name"), *(profile.get("brand_aliases") or [])]
        if value and len(str(value).strip()) >= 2
    ]
    for item in unbranded:
        text = str(item.get("text") or "")
        if any(_prompt_contains_alias(text, alias) for alias in aliases):
            errors.append(
                f"Безбрендовый сценарий {item.get('prompt_key')} содержит название цели."
            )
        if item.get("choice_request") is not True:
            errors.append(
                f"Безбрендовый сценарий {item.get('prompt_key')} не просит "
                "назвать конкретные варианты."
            )
    for item in branded:
        text = str(item.get("text") or "")
        if not aliases or not any(
            _prompt_contains_alias(text, alias) for alias in aliases
        ):
            errors.append(
                f"Брендовый сценарий {item.get('prompt_key')} не содержит "
                "подтверждённое название бренда или его алиас."
            )
    keys = [str(item.get("prompt_key") or "") for item in prompts]
    if len(keys) != len(set(keys)) or any(not key for key in keys):
        errors.append("Ключи сценариев должны быть непустыми и уникальными.")
    texts = [str(item.get("text") or "").strip().casefold() for item in prompts]
    if len(texts) != len(set(texts)) or any(not text for text in texts):
        errors.append("Тексты сценариев должны быть непустыми и уникальными.")
    if any(not str(item.get("rationale") or "").strip() for item in prompts):
        errors.append("Каждый сценарий должен объяснять, какой сигнал он проверяет.")
    return errors


def _prompt_review_errors(
    review: dict[str, Any],
    prompt_set: dict[str, Any],
    market_research: dict[str, Any] | None = None,
) -> list[str]:
    """Turn a model review into a strict, bounded acceptance decision."""

    research_citation_urls = {
        normalized
        for item in (
            (
                market_research.get("web_evidence", {}).get("citations", [])
                if isinstance(market_research, dict)
                and isinstance(market_research.get("web_evidence"), dict)
                else []
            )
        )
        if isinstance(item, dict)
        and (normalized := _normalized_research_url(item.get("url")))
    }
    # Сценарий может законно опираться на сам сайт (календарь событий,
    # вебинары): требовать внешнюю поисковую цитату для такого граундинга —
    # строже, чем гейт исследования, который эти site-URL уже сверил с реально
    # скачанными страницами. Прогон 5ae13350 (2026-08-21) падал ровно здесь:
    # критик одобрял transactional/nav сценарии с «Не требуется», а
    # пересечение только с веб-цитатами оставалось пустым.
    if isinstance(market_research, dict):
        site_confirmed = market_research.get("site_confirmed")
        if isinstance(site_confirmed, dict):
            for item in _nonempty_research_items(site_confirmed.get("evidence")):
                if isinstance(item, dict) and (
                    normalized := _normalized_research_url(item.get("url"))
                ):
                    research_citation_urls.add(normalized)
    expected = {
        str(item.get("prompt_key") or ""): str(item.get("intent_class") or "")
        for item in prompt_set.get("prompts") or []
        if item.get("role") == "unbranded_discovery"
    }
    checks = review.get("checks")
    if not isinstance(checks, list):
        return [_REVIEW_STRUCTURALLY_INVALID]

    by_key: dict[str, dict[str, Any]] = {}
    for check in checks:
        if not isinstance(check, dict):
            continue
        key = str(check.get("prompt_key") or "")
        if key in by_key:
            return [_REVIEW_STRUCTURALLY_INVALID]
        by_key[key] = check
    if set(by_key) != set(expected):
        return [_REVIEW_STRUCTURALLY_INVALID]

    errors: list[str] = []
    for key, declared_intent in expected.items():
        check = by_key[key]
        reviewed_declared = str(check.get("declared_intent") or "")
        dominant = str(check.get("dominant_intent") or "")
        matches = check.get("matches") is True
        if (
            reviewed_declared != declared_intent
            or dominant != declared_intent
            or not matches
        ):
            instruction = str(check.get("fix_instruction") or "").strip()
            reason = str(check.get("reason") or "").strip()
            errors.append(
                f"{key} должен соответствовать классу {declared_intent}: "
                f"{instruction or reason or 'переформулируйте пользовательскую задачу'}"
            )
        unsupported = [
            str(item).strip()
            for item in check.get("unsupported_assumptions") or []
            if str(item).strip()
        ]
        supporting_evidence = [
            str(item).strip()
            for item in check.get("supporting_evidence") or []
            if str(item).strip()
        ]
        supporting_urls: set[str] = set()
        for item in supporting_evidence:
            supporting_urls |= _urls_in_evidence_text(item)
        if (
            check.get("grounded_in_research") is not True
            or unsupported
            or not supporting_evidence
            or (
                market_research is not None
                and not bool(supporting_urls & research_citation_urls)
            )
        ):
            instruction = str(check.get("fix_instruction") or "").strip()
            errors.append(
                f"{key} содержит неподтверждённое рыночное допущение: "
                f"{instruction or '; '.join(unsupported) or 'свяжите сценарий с доказательствами исследования'}"
            )
    if review.get("verdict") != "pass" and not errors:
        errors.append(
            str(review.get("summary") or "").strip()
            or "Критик просит переработать INTENT-сценарии."
        )
    return errors


async def _review_prompt_set_semantics(
    run_id: str,
    profile: dict[str, Any],
    prompt_set: dict[str, Any],
    market_research: dict[str, Any],
) -> list[str]:
    """Check that each label describes the prompt's actual dominant intent."""

    research = _require_market_research_usable(market_research)
    research_digest = _stable_json_sha256(research)
    prompts = [
        item
        for item in prompt_set.get("prompts") or []
        if item.get("role") == "unbranded_discovery"
    ]
    payload = {
        "site_profile": {
            key: profile.get(key)
            for key in (
                "brand_name",
                "brand_aliases",
                "category",
                "market",
                "audiences",
                "customer_jobs",
                "decision_criteria",
                "geography",
            )
        },
        "market_research": research,
        "market_research_digest": research_digest,
        "canonical_intent_definitions": INTENT_DEFINITIONS,
        "prompts": prompts,
        "prompt_keys_to_check": [
            str(item.get("prompt_key") or "") for item in prompts
        ],
    }
    system = f"""
Ты независимый методолог-критик экспресс-исследования AI visibility.
Проверь шесть безбрендовых пользовательских запросов по канонической модели
INTENT. Классы нельзя переопределять:

I — Information Seeking: общие сведения и понимание темы.
E — Evaluative: сравнение вариантов и критериев выбора.
T — Transactional: готовность купить, заказать или принять решение.
NB — Need Based: задача, боль, ограничение или контекст использования.
NAV — Navigation: источник, площадка, обзор, агрегатор или точка входа.
TR — Trend-Driven: тренды, новизна, популярность или меняющееся поведение.

Для каждого запроса определи одно доминирующее намерение. NB нельзя принимать
за навигацию, NAV — за общий поиск типа решения, TR — за доверие или проверку
риска. Просьба назвать конкретные варианты есть во всех сценариях как
измерительная рамка; сама по себе она не превращает каждый запрос в E.

В checks верни ровно шесть объектов — по одному на каждый prompt_key из
prompt_keys_to_check, без пропусков и добавлений.

Поставь pass, только если все шесть запросов естественны для обычного
пользователя, различаются по доминирующему намерению, соответствуют заявленным
классам и способны вызвать конкретные названия компаний, продуктов, сервисов
или источников. Дополнительно проверь, что аудитория, задача, критерий выбора,
география и терминология каждого запроса прямо подтверждены market_research:
используй его evidence, citations и confidence. Основной бренд бери только из
site_profile. Если сценарий опирается на неподтверждённое допущение, поставь
grounded_in_research=false, перечисли unsupported_assumptions и потребуй
исправления. В supporting_evidence укажи конкретные подтверждения и URL.
Иначе поставь revise и дай короткую точную инструкцию для исправления каждого
несовпавшего запроса. Не переписывай всю методологию.

{LIVE_RUSSIAN_RULES}
""".strip()
    errors: list[str] = []
    for review_attempt in range(3):
        attempt_payload = dict(payload)
        if review_attempt:
            # Кэш-бастер: _structured_artifact кэширует по input_json, и без
            # изменения входа повтор вернул бы тот же дефектный вердикт.
            attempt_payload["review_attempt"] = review_attempt
            attempt_payload["previous_review_error"] = (
                _REVIEW_STRUCTURALLY_INVALID
            )
        review = await _structured_artifact(
            run_id,
            stage_key="scenario_design",
            artifact_key="prompt_set_semantic_review",
            schema=PROMPT_SET_REVIEW_SCHEMA,
            schema_name="aiv_prompt_set_semantic_review",
            system=system,
            user_payload=attempt_payload,
            max_tokens=6000,
            # Не CRITIC_MODEL: gemini-flash игнорирует minItems строгой схемы
            # и явное требование «ровно шесть проверок» — возвращал 1-2
            # проверки с verdict=pass (прогон 5ae13350, 2026-08-21).
            # Processing-модель надёжнее, но и она изредка теряет элемент,
            # поэтому структурно дефектное ревью повторяется здесь же.
            model=PROCESSING_MODEL,
            reasoning_effort="high",
            prompt_version=PROMPT_SET_REVIEW_VERSION,
        )
        errors = _prompt_review_errors(review, prompt_set, research)
        if errors != [_REVIEW_STRUCTURALLY_INVALID]:
            break
    return errors


async def _generate_prompt_set(
    run_id: str,
    profile: dict[str, Any],
    requested_site: dict[str, Any] | None = None,
    *,
    market_research: dict[str, Any],
) -> dict[str, Any]:
    research = _require_market_research_usable(market_research)
    payload: dict[str, Any] = {
        "requested_site": requested_site or {},
        "site_profile": profile,
        "market_research": research,
        "market_research_digest": _stable_json_sha256(research),
    }
    cached = await _artifact_output(
        run_id,
        "prompt_set",
        input_json=payload,
        model=ANALYSIS_MODEL,
        prompt_version=PROMPT_SET_VERSION,
    )
    last_errors: list[str] = []
    if isinstance(cached, dict):
        last_errors = _validate_prompt_set(cached, profile)
        if not last_errors:
            last_errors = await _review_prompt_set_semantics(
                run_id,
                profile,
                cached,
                research,
            )
        if not last_errors:
            return cached
    system = f"""
Ты проектируешь экспресс-проверку AI visibility для одного бренда.

Создай ровно девять естественных пользовательских запросов:
1. Шесть безбрендовых запросов для discovery. Каждый покрывает ровно один
INTENT-класс по его каноническому смыслу:
- I — Information Seeking: общие сведения и понимание темы;
- E — Evaluative: сравнение вариантов и критериев выбора;
- T — Transactional: готовность купить, заказать или принять решение;
- NB — Need Based: задача, боль, ограничение или контекст использования;
- NAV — Navigation: источник, площадка, обзор, агрегатор или точка входа;
- TR — Trend-Driven: тренды, новизна, популярность или меняющееся поведение.
NB нельзя подменять навигацией, NAV — общим поиском типа решения, а TR —
доверием или проверкой риска. В этих запросах нельзя называть целевой бренд,
его алиасы и фирменные названия продуктов.
2. Три брендовых запроса от пользователя, который уже знает бренд:
проверка предложения, сравнение или доверие. Они не входят в долю
безбрендового обнаружения и имеют отдельный role. В каждом таком запросе
обязательно прямо назови основной бренд или подтверждённый алиас из профиля:
иначе это не диагностический запрос знающего пользователя.

Запросы должны соответствовать реальной категории, аудитории и географии.
Строй их из customer_jobs и decision_criteria аттестованного market_research:
так, как человек формулирует вопрос до покупки, сравнения или проверки
поставщика. Используй только подтверждённые evidence и citations с достаточной
confidence. Не превращай неподтверждённое предположение в пользовательский
сценарий.

Каждая конкретная деталь в тексте запроса — перечень направлений, форматы,
условия, сроки, слова «сейчас» или «в этом году» — обязана быть прямо
подтверждена market_research или site_profile. Если подтверждения нет,
формулируй запрос нейтрально, без этой детали: естественный общий вопрос
лучше конкретики без источника, потому что критик отклонит её как
неподтверждённое допущение. Основной бренд, алиасы и портфель бери только из site_profile:
внешнее исследование не вправе переопределять цель.
Не подсказывай моделям желаемый ответ и не перечисляй конкурентов заранее.
Каждый запрос проверяет одну задачу выбора, без склейки нескольких INTENT.

Критически важно: каждый из шести безбрендовых запросов должен естественно
просить назвать конкретные компании, продукты, сервисы, источники или варианты
выбора. Абстрактный вопрос «как устроено» или «на что смотреть» без просьбы
назвать варианты не измеряет обнаружение бренда и не подходит. Поле
choice_request поставь true только если текст действительно способен вызвать
перечень названных решений. Для информационного INTENT сначала задай полезный
контекст, а затем попроси привести конкретные примеры или поставщиков.

В rationale объясни, какой именно признак заявленного INTENT-класса выражен
в формулировке. Не используй код класса как формальное оправдание.

{LIVE_RUSSIAN_RULES}
""".strip()
    # Четыре, а не две: строгий семантический критик (все шесть проверок,
    # processing-модель) обычно требует 2-3 итерации на сходимость — с двумя
    # попытками прогон profi.travel умирал на одном несведённом TR-сценарии.
    previous_set: dict[str, Any] | None = None
    for attempt in range(4):
        user_content = dict(payload)
        if last_errors:
            user_content["validation_errors_to_fix"] = last_errors
        if last_errors and previous_set is not None:
            # Точечный ремонт: полная регенерация с нуля не сходится — каждая
            # новая редакция набора рождает новые формулировки и новые
            # замечания критика. Сценарии без замечаний фиксируются дословно.
            user_content["previous_prompt_set"] = previous_set
            user_content["repair_instruction"] = (
                "Исправь только сценарии, упомянутые в "
                "validation_errors_to_fix. Остальные сценарии из "
                "previous_prompt_set перенеси в ответ дословно, не меняя ни "
                "текст, ни prompt_key."
            )
        result = await chat(
            model=ANALYSIS_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user_content, ensure_ascii=False)},
            ],
            response_schema=PROMPT_SET_SCHEMA,
            schema_name="aiv_prompt_set",
            reasoning_effort="high",
            max_tokens=7000,
            temperature=0.2,
        )
        if not isinstance(result.parsed, dict):
            last_errors = ["Ответ не является объектом."]
            continue
        previous_set = result.parsed
        last_errors = _validate_prompt_set(result.parsed, profile)
        if not last_errors:
            last_errors = await _review_prompt_set_semantics(
                run_id,
                profile,
                result.parsed,
                research,
            )
        if not last_errors:
            await _save_artifact(
                run_id,
                stage_key="scenario_design",
                artifact_key="prompt_set",
                status="completed",
                model=ANALYSIS_MODEL,
                input_json=payload,
                output_json=result.parsed,
                raw_text=result.text,
                usage_json=result.usage,
                prompt_version=PROMPT_SET_VERSION,
            )
            return result.parsed
    raise OpenRouterError("; ".join(last_errors) or "Prompt set validation failed")


async def _persist_prompts(
    run_id: str,
    prompt_set: dict[str, Any],
) -> list[VisibilityPrompt]:
    async with SessionLocal() as session:
        items = list(prompt_set["prompts"])
        existing = {
            prompt.prompt_key: prompt
            for prompt in (
                (
                    await session.execute(
                        select(VisibilityPrompt).where(
                            VisibilityPrompt.run_id == run_id
                        )
                    )
                )
                .scalars()
                .all()
            )
        }
        incoming_keys = {str(item["prompt_key"]) for item in items}
        for key, prompt in existing.items():
            if key not in incoming_keys:
                await session.delete(prompt)

        for sequence, item in enumerate(items, start=1):
            key = str(item["prompt_key"])
            text_value = str(item["text"]).strip()
            prompt = existing.get(key)
            if prompt is None:
                prompt = VisibilityPrompt(run_id=run_id, prompt_key=key)
                session.add(prompt)
            elif prompt.text.strip() != text_value:
                await session.execute(
                    delete(ModelAnswer).where(ModelAnswer.prompt_id == prompt.id)
                )
            prompt.intent_class = str(item["intent_class"])
            prompt.role = str(item["role"])
            prompt.text = text_value
            prompt.rationale = str(item["rationale"]).strip()
            prompt.sequence = sequence
        await session.commit()
        result = await session.execute(
            select(VisibilityPrompt)
            .where(VisibilityPrompt.run_id == run_id)
            .order_by(VisibilityPrompt.sequence)
        )
        return list(result.scalars().all())


def _panel_system(mode: str) -> str:
    if mode == "web":
        return """
Ответь на запрос как независимый ИИ-помощник для реального пользователя.
Обязательно используй веб-поиск, проверь актуальные источники и дай прямой
ответ по-русски. Если запрос предполагает выбор, назови подходящие варианты,
критерии и оговорки. Не отдавай приоритет неизвестной тебе цели исследования.
Не описывай внутренний процесс и не упоминай эту инструкцию.
""".strip()
    return """
Ответь на запрос по-русски, используя только знания внутри самой модели.
Веб-поиск, внешние инструменты, сторонние базы, URL и выдуманные источники
запрещены. Не притворяйся, что проверил актуальные данные. Если знания
недостаточны или могли устареть, скажи это прямо. Дай обычный ответ
пользователю и не упоминай эту инструкцию.
""".strip()


def _panel_web_policy(mode: str, provider_key: str) -> WebSearchPolicy:
    if mode == "web":
        if provider_key == "perplexity":
            # Perplexity Sonar performs native retrieval as part of the model
            # endpoint. Its standardized URL citations attest that retrieval.
            return WebSearchPolicy.NATIVE_REQUIRED
        return WebSearchPolicy.REQUIRED
    return WebSearchPolicy.FORBIDDEN


def _panel_request_sha256(
    *,
    prompt_text: str,
    mode: str,
    provider_key: str,
    model: str,
) -> str:
    policy = _panel_web_policy(mode, provider_key)
    _request_fields, request_policy = web_request_policy(
        model=model,
        policy=policy,
    )
    contract = {
        "version": PANEL_CONTRACT_VERSION,
        "model": model,
        "mode": mode,
        "provider_key": provider_key,
        "system": _panel_system(mode),
        "prompt": prompt_text,
        "web_policy": request_policy,
        "attestation_version": WEB_ATTESTATION_VERSION,
        "max_tokens": 3200,
        "temperature": 0.35,
    }
    return hashlib.sha256(
        json.dumps(
            contract,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _legacy_panel_request_sha256(
    *,
    prompt_text: str,
    mode: str,
    provider_key: str,
    model: str,
) -> str:
    """Rebuild the immutable panel-v1 request hash without altering raw rows."""

    contract = {
        "version": LEGACY_PANEL_CONTRACT_VERSION,
        "model": model,
        "mode": mode,
        "provider_key": provider_key,
        "system": _panel_system(mode),
        "prompt": prompt_text,
        "web_search": mode == "web" and provider_key != "perplexity",
        "max_tokens": 3200,
        "temperature": 0.35,
    }
    return hashlib.sha256(
        json.dumps(
            contract,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _legacy_usage_count(usage: dict[str, Any], key: str) -> int:
    """Read the telemetry shape used before the v2 attestation contract."""

    values: list[Any] = []
    for container_key in ("server_tool_use_details", "server_tool_use"):
        container = usage.get(container_key)
        if isinstance(container, dict):
            values.append(container.get(key))
    return max(
        (
            value
            for value in values
            if isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 0
        ),
        default=0,
    )


_LEGACY_RETRIEVAL_COUNTER_KEYS = frozenset(
    {
        "web_search_requests",
        "web_fetch_requests",
        "search_requests",
        "fetch_requests",
        "browser_requests",
        "tool_calls_requested",
        "tool_calls_executed",
    }
)
_LEGACY_RETRIEVAL_KEY_FRAGMENTS = (
    "web_search",
    "web_fetch",
    "tool_call",
    "browser_request",
    "retrieval_request",
)


def _legacy_memory_trace_reasons(answer: ModelAnswer) -> list[str]:
    """Return every persisted signal incompatible with an offline observation.

    The legacy transport did not persist a complete request-policy bundle, so
    absence is accepted only for a frozen, whole-cohort historical analysis.
    Any malformed or vendor-specific retrieval trace fails closed; a boolean
    or string ``0`` is not accepted as a trustworthy counter.
    """

    reasons: list[str] = []
    citations = answer.citations_json
    if citations not in (None, []):
        reasons.append("citations_present")

    usage = answer.usage_json if isinstance(answer.usage_json, dict) else {}

    def walk(value: Any, path: tuple[str, ...] = ()) -> None:
        if isinstance(value, dict):
            for raw_key, child in value.items():
                key = str(raw_key).casefold()
                child_path = (*path, key)
                # A validated v1 request hash is checked separately.  It is
                # provenance, not response-side retrieval telemetry.
                if child_path == ("_aiv_panel_contract",):
                    continue
                if key in _LEGACY_RETRIEVAL_COUNTER_KEYS:
                    if not (
                        isinstance(child, int)
                        and not isinstance(child, bool)
                        and child == 0
                    ):
                        reasons.append("usage." + ".".join(child_path))
                    continue
                if any(
                    fragment in key
                    for fragment in _LEGACY_RETRIEVAL_KEY_FRAGMENTS
                ):
                    if child not in (None, [], {}):
                        reasons.append("usage." + ".".join(child_path))
                    continue
                if key in {
                    "tools",
                    "tool_calls",
                    "plugins",
                    "annotations",
                    "citations",
                    "sources",
                    "router_signals",
                    "router_metadata",
                    "_aiv_router_metadata",
                    "_aiv_response_annotations",
                    "_aiv_request_policy",
                    "_aiv_web_attestation",
                }:
                    if child not in (None, [], {}):
                        reasons.append("usage." + ".".join(child_path))
                    continue
                walk(child, child_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, (*path, str(index)))

    walk(usage)
    return list(dict.fromkeys(reasons))


def _legacy_panel_answer_attestation(
    answer: ModelAnswer,
    *,
    prompt_text: str,
    provenance: dict[str, Any] | None,
) -> tuple[bool, str]:
    """Qualify immutable legacy evidence without pretending it is panel-v2.

    A saved mode remains a declaration made by the historical pipeline. Web
    rows additionally need positive retrieval evidence. Memory rows are kept
    as a clearly labelled legacy declaration only when no retrieval trace is
    present. Nothing is written back to ``usage_json``.
    """

    if answer.mode not in {"web", "memory"}:
        return False, "legacy_mode_unknown"
    expected_model = LEGACY_PANEL_MODELS.get(answer.provider_key)
    model = str(answer.model or "").casefold()
    if expected_model is None or model != expected_model:
        return False, "legacy_provider_model_mismatch"

    if provenance is not None:
        if provenance.get("version") != LEGACY_PANEL_CONTRACT_VERSION:
            return False, "stale_panel_contract"
        if provenance.get("request_sha256") != _legacy_panel_request_sha256(
            prompt_text=prompt_text,
            mode=answer.mode,
            provider_key=answer.provider_key,
            model=answer.model,
        ):
            return False, "legacy_request_hash_mismatch"

    usage = answer.usage_json if isinstance(answer.usage_json, dict) else {}
    citation_count = sum(
        1
        for citation in (answer.citations_json or [])
        if isinstance(citation, dict)
        and urlparse(str(citation.get("url") or "")).scheme in {"http", "https"}
        and bool(urlparse(str(citation.get("url") or "")).netloc)
    )
    web_search_requests = _legacy_usage_count(
        usage,
        "web_search_requests",
    )
    web_fetch_requests = _legacy_usage_count(
        usage,
        "web_fetch_requests",
    )
    tool_calls_requested = _legacy_usage_count(
        usage,
        "tool_calls_requested",
    )
    tool_calls_executed = _legacy_usage_count(
        usage,
        "tool_calls_executed",
    )

    if answer.mode == "web":
        if answer.provider_key == "perplexity":
            if citation_count < 1:
                return False, "legacy_native_web_evidence_missing"
        elif web_search_requests < 1 or citation_count < 1:
            return False, "legacy_web_evidence_missing"
        return True, "legacy_web_retrieval_confirmed"

    if ":online" in model:
        return False, "legacy_online_variant_in_memory_mode"
    if _legacy_memory_trace_reasons(answer) or any(
        (
            citation_count,
            web_search_requests,
            web_fetch_requests,
            tool_calls_requested,
            tool_calls_executed,
        )
    ):
        return False, "legacy_memory_retrieval_observed"
    return False, "legacy_memory_request_not_enforced"


def _panel_answer_attestation(
    answer: ModelAnswer,
    *,
    prompt_text: str,
    legacy_allowed: bool = False,
) -> tuple[bool, str]:
    """Fail closed unless saved usage proves the requested web-access mode."""

    usage = answer.usage_json if isinstance(answer.usage_json, dict) else {}
    provenance = usage.get("_aiv_panel_contract")
    request_policy = usage.get("_aiv_request_policy")
    attestation = usage.get("_aiv_web_attestation")
    annotations = usage.get("_aiv_response_annotations")
    strict_bundle_present = any(
        value is not None
        for value in (
            provenance,
            request_policy,
            attestation,
            annotations,
        )
    )
    if not strict_bundle_present:
        if not legacy_allowed:
            return False, "legacy_run_contract_unverified"
        return _legacy_panel_answer_attestation(
            answer,
            prompt_text=prompt_text,
            provenance=None,
        )
    if (
        isinstance(provenance, dict)
        and provenance.get("version") == LEGACY_PANEL_CONTRACT_VERSION
        and request_policy is None
        and attestation is None
        and annotations is None
    ):
        if not legacy_allowed:
            return False, "legacy_run_contract_unverified"
        return _legacy_panel_answer_attestation(
            answer,
            prompt_text=prompt_text,
            provenance=provenance,
        )
    if not isinstance(provenance, dict):
        return False, "missing_panel_contract"
    if not isinstance(request_policy, dict):
        return False, "missing_request_policy"
    if not isinstance(attestation, dict):
        return False, "missing_web_attestation"
    if not isinstance(annotations, list):
        return False, "missing_response_annotations"

    expected_policy = _panel_web_policy(
        answer.mode,
        answer.provider_key,
    ).value
    try:
        _expected_fields, expected_request_policy = web_request_policy(
            model=answer.model,
            policy=expected_policy,
        )
        expected_request_sha256 = _panel_request_sha256(
            prompt_text=prompt_text,
            mode=answer.mode,
            provider_key=answer.provider_key,
            model=answer.model,
        )
    except OpenRouterError:
        # Legacy evidence may contain deprecated ``:online`` slugs or a
        # non-Sonar Perplexity model. It must remain stored and visible, but
        # cannot abort a rebuild or enter current metrics.
        return False, "unsupported_legacy_web_contract"
    if provenance.get("version") != PANEL_CONTRACT_VERSION:
        return False, "stale_panel_contract"
    if provenance.get("request_sha256") != expected_request_sha256:
        return False, "request_hash_mismatch"
    if provenance.get("attestation_version") != WEB_ATTESTATION_VERSION:
        return False, "stale_attestation_contract"
    if request_policy.get("version") != WEB_ATTESTATION_VERSION:
        return False, "stale_request_policy"
    if request_policy.get("policy") != expected_policy:
        return False, "request_policy_mismatch"
    stored_policy_contract = {
        key: value
        for key, value in request_policy.items()
        if key != "sha256"
    }
    stored_policy_sha256 = hashlib.sha256(
        json.dumps(
            stored_policy_contract,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if request_policy.get("sha256") != stored_policy_sha256:
        return False, "corrupt_request_policy_hash"
    if request_policy.get("sha256") != expected_request_policy.get("sha256"):
        return False, "request_policy_contract_mismatch"
    if provenance.get("request_policy_sha256") != request_policy.get("sha256"):
        return False, "request_policy_hash_mismatch"
    if provenance.get("web_attestation") != attestation:
        return False, "attestation_provenance_mismatch"
    if attestation.get("version") != WEB_ATTESTATION_VERSION:
        return False, "stale_web_attestation"
    if attestation.get("policy") != expected_policy:
        return False, "attestation_policy_mismatch"
    if (
        attestation.get("state") != "verified"
        or attestation.get("metric_eligible") is not True
        or attestation.get("violations")
    ):
        return False, "web_policy_not_verified"

    def count(value: Any) -> int:
        return (
            value
            if isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 0
            else 0
        )

    citation_count = len(answer.citations_json or [])
    citation_annotation_count = sum(
        bool(
            isinstance(item, dict)
            and isinstance(item.get("url_citation"), dict)
            and str((item.get("url_citation") or {}).get("url") or "").strip()
        )
        for item in annotations
    )
    if expected_policy == WebSearchPolicy.REQUIRED.value:
        if count(attestation.get("web_search_requests")) < 1:
            return False, "web_search_not_observed"
        if citation_count < 1 or citation_annotation_count < 1:
            return False, "web_citations_not_observed"
    elif expected_policy == WebSearchPolicy.NATIVE_REQUIRED.value:
        response_model = str(attestation.get("response_model") or "")
        if (
            not response_model.casefold().startswith("perplexity/sonar")
            or citation_count < 1
            or citation_annotation_count < 1
        ):
            return False, "perplexity_native_search_not_observed"
    else:
        request_fields = request_policy.get("request_fields")
        if not isinstance(request_fields, dict):
            return False, "missing_forbidden_request_fields"
        plugins = request_fields.get("plugins")
        if plugins != [{"id": "web", "enabled": False}]:
            return False, "web_plugin_not_explicitly_disabled"
        if request_fields.get("tool_choice") != "none":
            return False, "tool_choice_not_none"
        if request_fields.get("tools"):
            return False, "tools_present_while_forbidden"
        if ":online" in answer.model.casefold():
            return False, "online_variant_while_forbidden"
        if (
            count(attestation.get("web_search_requests")) > 0
            or count(attestation.get("web_fetch_requests")) > 0
            or citation_count > 0
            or annotations
            or attestation.get("router_signals")
        ):
            return False, "retrieval_observed_while_forbidden"
    return True, "verified"


def _panel_metric_access(
    answer: ModelAnswer,
    *,
    transport_attested: bool,
    attestation_reason: str,
    legacy_memory_observation_allowed: bool,
) -> dict[str, Any]:
    """Separate aggregate eligibility from raw-context eligibility.

    Current panel calls prove their requested transport policy and may be used
    both in metrics and as full-text narrative evidence.  Historical memory
    calls predate the persisted request-policy bundle.  A validated legacy
    run can still provide a bounded observation when the exact configured
    offline model was used and no citation, web-search, fetch or tool signal
    was observed.  Those rows may enter deterministic aggregates, but remain
    explicitly limited and their raw text is withheld from the final writer.
    """

    legacy_memory_observation = bool(
        legacy_memory_observation_allowed
        and not transport_attested
        and answer.mode == "memory"
        and attestation_reason == LEGACY_MEMORY_OBSERVATION_REASON
    )
    if transport_attested:
        evidence_state = (
            "strict_verified"
            if attestation_reason == "verified"
            else "legacy_retrieval_confirmed"
        )
    elif legacy_memory_observation:
        evidence_state = "legacy_observational"
    else:
        evidence_state = "excluded"
    return {
        "metric_eligible": bool(
            transport_attested or legacy_memory_observation
        ),
        "context_eligible": bool(transport_attested),
        "metric_evidence_state": evidence_state,
        "metric_limitation": (
            LEGACY_MEMORY_OBSERVATION_REASON
            if legacy_memory_observation
            else None
        ),
    }


async def _legacy_panel_run_contract(run_id: str) -> dict[str, Any]:
    """Validate the historical run as a whole before any legacy fallback."""

    async with SessionLocal() as session:
        run = (
            await session.execute(select(Run).where(Run.id == run_id))
        ).scalar_one_or_none()
        prompt_artifact = (
            await session.execute(
                select(RunArtifact).where(
                    RunArtifact.run_id == run_id,
                    RunArtifact.artifact_key == "prompt_set",
                )
            )
        ).scalar_one_or_none()
        prompts = list(
            (
                await session.execute(
                    select(VisibilityPrompt)
                    .where(VisibilityPrompt.run_id == run_id)
                    .order_by(VisibilityPrompt.sequence, VisibilityPrompt.id)
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

    reasons: list[str] = []
    config = run.config_json if run is not None else {}
    if not isinstance(config, dict) or config.get("pipeline_version") != LEGACY_PIPELINE_VERSION:
        reasons.append("pipeline_version_mismatch")
    if prompt_artifact is None or prompt_artifact.status != "completed":
        reasons.append("prompt_set_not_completed")
    elif prompt_artifact.prompt_version not in LEGACY_PROMPT_SET_VERSIONS:
        reasons.append("prompt_set_version_not_allowed")

    artifact_output = (
        prompt_artifact.output_json
        if prompt_artifact is not None
        and isinstance(prompt_artifact.output_json, dict)
        else {}
    )
    artifact_prompts = artifact_output.get("prompts")
    if not isinstance(artifact_prompts, list) or not artifact_prompts:
        reasons.append("prompt_set_output_missing")
        artifact_prompts = []

    persisted = [
        {
            "prompt_key": prompt.prompt_key,
            "intent_class": prompt.intent_class,
            "role": prompt.role,
            "text": prompt.text,
            "rationale": prompt.rationale,
        }
        for prompt in prompts
    ]
    artifact = [
        {
            "prompt_key": item.get("prompt_key"),
            "intent_class": item.get("intent_class"),
            "role": item.get("role"),
            "text": item.get("text"),
            "rationale": item.get("rationale"),
        }
        for item in artifact_prompts
        if isinstance(item, dict)
    ]
    if len(prompts) != 9 or len(artifact) != 9:
        reasons.append("legacy_prompt_count_mismatch")
    if sorted(prompt.sequence for prompt in prompts) != list(range(1, 10)):
        reasons.append("legacy_prompt_sequence_mismatch")
    if persisted != artifact:
        reasons.append("persisted_prompt_set_mismatch")
    if len({item["prompt_key"] for item in persisted}) != len(persisted):
        reasons.append("duplicate_prompt_key")

    prompt_text_by_id = {prompt.id: prompt.text for prompt in prompts}
    expected_memory_cells = {
        (prompt.id, provider_key)
        for prompt in prompts
        for provider_key in LEGACY_MEMORY_MODELS
    }
    memory_answers = [answer for answer in answers if answer.mode == "memory"]
    actual_memory_cells = [
        (answer.prompt_id, answer.provider_key) for answer in memory_answers
    ]
    memory_reasons: list[str] = []
    if len(memory_answers) != len(expected_memory_cells):
        memory_reasons.append("legacy_memory_cell_count_mismatch")
    if set(actual_memory_cells) != expected_memory_cells:
        memory_reasons.append("legacy_memory_cell_grid_mismatch")
    if len(set(actual_memory_cells)) != len(actual_memory_cells):
        memory_reasons.append("legacy_memory_duplicate_cell")

    memory_signatures: list[dict[str, Any]] = []
    for answer in memory_answers:
        expected_model = LEGACY_MEMORY_MODELS.get(answer.provider_key)
        prompt_text = prompt_text_by_id.get(answer.prompt_id)
        row_reasons: list[str] = []
        if prompt_text is None:
            row_reasons.append("unknown_prompt")
        if expected_model is None:
            row_reasons.append("unexpected_provider")
        elif str(answer.model or "").casefold() != expected_model:
            row_reasons.append("model_mismatch")
        if answer.status != "completed" or not str(
            answer.response_text or ""
        ).strip():
            row_reasons.append("answer_incomplete")
        if ":online" in str(answer.model or "").casefold():
            row_reasons.append("online_variant")
        row_reasons.extend(_legacy_memory_trace_reasons(answer))

        usage = answer.usage_json if isinstance(answer.usage_json, dict) else {}
        provenance = usage.get("_aiv_panel_contract")
        if provenance is not None:
            if not isinstance(provenance, dict) or prompt_text is None:
                row_reasons.append("invalid_panel_v1_provenance")
            else:
                _verified, attestation_reason = (
                    _legacy_panel_answer_attestation(
                        answer,
                        prompt_text=prompt_text,
                        provenance=provenance,
                    )
                )
                if attestation_reason != LEGACY_MEMORY_OBSERVATION_REASON:
                    row_reasons.append(attestation_reason)

        memory_signatures.append(
            {
                "answer_id": answer.id,
                "prompt_id": answer.prompt_id,
                "provider_key": answer.provider_key,
                "model": answer.model,
                "status": answer.status,
                "raw_answer_sha256": hashlib.sha256(
                    str(answer.response_text or "").encode("utf-8")
                ).hexdigest(),
                "citations_sha256": _stable_json_sha256(
                    answer.citations_json or []
                ),
                "usage_sha256": _stable_json_sha256(usage),
                "reasons": sorted(set(row_reasons)),
            }
        )
        memory_reasons.extend(row_reasons)

    memory_reasons = list(dict.fromkeys(memory_reasons))
    memory_observation_eligible = bool(
        not reasons
        and not memory_reasons
        and len(memory_answers) == len(expected_memory_cells)
    )

    contract = {
        "version": LEGACY_PANEL_EVIDENCE_VERSION,
        "pipeline_version": config.get("pipeline_version") if isinstance(config, dict) else None,
        "prompt_set_version": (
            prompt_artifact.prompt_version if prompt_artifact is not None else None
        ),
        "persisted_prompts_sha256": _stable_json_sha256(persisted),
        "artifact_prompts_sha256": _stable_json_sha256(artifact),
        "eligible": not reasons,
        "reasons": reasons,
        "memory_observation_eligible": memory_observation_eligible,
        "memory_observation_reasons": memory_reasons,
        "memory_cell_count": len(memory_answers),
        "expected_memory_cell_count": len(expected_memory_cells),
        "memory_manifest_sha256": _stable_json_sha256(memory_signatures),
    }
    return {**contract, "digest": _stable_json_sha256(contract)}


def _panel_evidence_sha256(
    answer: ModelAnswer,
    *,
    reason: str,
    legacy_contract: dict[str, Any],
) -> str:
    return _stable_json_sha256(
        {
            "reason": reason,
            "usage": answer.usage_json or {},
            "citations": answer.citations_json or [],
            "legacy_run_contract_digest": (
                legacy_contract.get("digest")
                if reason.startswith("legacy_")
                else None
            ),
        }
    )


async def _ensure_answer_rows(
    run_id: str,
    prompts: list[VisibilityPrompt],
    mode: str,
) -> list[tuple[int, str, str, str, str, str]]:
    jobs: list[tuple[int, str, str, str, str, str]] = []
    async with SessionLocal() as session:
        existing_rows = list(
            (
                await session.execute(
                    select(ModelAnswer).where(
                        ModelAnswer.run_id == run_id,
                        ModelAnswer.mode == mode,
                    )
                )
            )
            .scalars()
            .all()
        )
        existing = {
            (row.prompt_id, row.provider_key): row for row in existing_rows
        }
        for prompt in prompts:
            for panel in panel_models():
                selected_model = panel.model if mode == "web" else panel.memory_model
                if selected_model is None:
                    continue
                request_sha256 = _panel_request_sha256(
                    prompt_text=prompt.text,
                    mode=mode,
                    provider_key=panel.key,
                    model=selected_model,
                )
                row = existing.get((prompt.id, panel.key))
                if row is None:
                    row = ModelAnswer(
                        run_id=run_id,
                        prompt_id=prompt.id,
                        provider_key=panel.key,
                        model=selected_model,
                        mode=mode,
                        status="pending",
                    )
                    session.add(row)
                    await session.flush()
                elif (
                    row.status == "completed"
                    and row.response_text
                ):
                    # Raw panel answers are append-only evidence. A legacy or
                    # stale completed cell remains visible for audit/rebuild,
                    # but _metric_rows will fail it closed unless its exact
                    # request and web policy are attested.
                    continue
                await session.execute(
                    delete(AnswerAnnotation).where(
                        AnswerAnnotation.answer_id == row.id
                    )
                )
                row.model = selected_model
                row.status = "pending"
                row.response_text = None
                row.citations_json = None
                row.usage_json = None
                row.error_message = None
                jobs.append(
                    (
                        row.id,
                        prompt.text,
                        panel.key,
                        panel.label,
                        selected_model,
                        request_sha256,
                    )
                )
        await session.commit()
    return jobs


async def _run_panel(
    run_id: str,
    prompts: list[VisibilityPrompt],
    *,
    mode: str,
    start_percent: int,
    end_percent: int,
) -> None:
    jobs = await _ensure_answer_rows(run_id, prompts, mode)
    async with SessionLocal() as session:
        existing_completed = len(
            (
                await session.execute(
                    select(ModelAnswer.id).where(
                        ModelAnswer.run_id == run_id,
                        ModelAnswer.mode == mode,
                        ModelAnswer.status == "completed",
                    )
                )
            ).all()
        )
        total_rows = len(
            (
                await session.execute(
                    select(ModelAnswer.id).where(
                        ModelAnswer.run_id == run_id,
                        ModelAnswer.mode == mode,
                    )
                )
            ).all()
        )
    if not jobs:
        return

    semaphore = asyncio.Semaphore(
        max(1, min(10, settings.OPENROUTER_PANEL_CONCURRENCY))
    )
    progress_lock = asyncio.Lock()
    completed_now = existing_completed

    async def worker(job: tuple[int, str, str, str, str, str]) -> None:
        nonlocal completed_now
        (
            answer_id,
            prompt_text,
            provider_key,
            _provider_label,
            model,
            request_sha256,
        ) = job

        def usage_with_provenance(result: Any) -> dict[str, Any]:
            usage = dict(result.usage or {})
            usage["_aiv_panel_contract"] = {
                "version": PANEL_CONTRACT_VERSION,
                "request_sha256": request_sha256,
                "request_policy_sha256": result.request_policy.get("sha256"),
                "web_policy": result.request_policy.get("policy"),
                "attestation_version": WEB_ATTESTATION_VERSION,
                "web_attestation": result.web_attestation,
            }
            return usage

        try:
            async with semaphore:
                result = await chat(
                    model=model,
                    messages=[
                        {"role": "system", "content": _panel_system(mode)},
                        {"role": "user", "content": prompt_text},
                    ],
                    web_policy=_panel_web_policy(mode, provider_key),
                    max_tokens=3200,
                    temperature=0.35,
                )
            async with SessionLocal() as session:
                row = (
                    await session.execute(
                        select(ModelAnswer).where(ModelAnswer.id == answer_id)
                    )
                ).scalar_one()
                row.status = "completed"
                row.response_text = result.text
                row.citations_json = result.citations or None
                row.usage_json = usage_with_provenance(result)
                row.error_message = None
                await session.commit()
        except asyncio.CancelledError:
            raise
        except OpenRouterPolicyError as exc:
            logger.warning(
                "Panel web-policy attestation failed for answer %s",
                answer_id,
            )
            async with SessionLocal() as session:
                row = (
                    await session.execute(
                        select(ModelAnswer).where(ModelAnswer.id == answer_id)
                    )
                ).scalar_one_or_none()
                if row is not None:
                    row.status = "failed"
                    row.response_text = exc.result.text
                    row.citations_json = exc.result.citations or None
                    row.usage_json = usage_with_provenance(exc.result)
                    row.error_message = str(exc)[:1000]
                    await session.commit()
        except Exception as exc:
            logger.warning(
                "Panel response failed for answer %s: %s",
                answer_id,
                type(exc).__name__,
            )
            async with SessionLocal() as session:
                row = (
                    await session.execute(
                        select(ModelAnswer).where(ModelAnswer.id == answer_id)
                    )
                ).scalar_one_or_none()
                if row is not None:
                    row.status = "failed"
                    row.error_message = str(exc)[:1000]
                    await session.commit()
        finally:
            async with progress_lock:
                completed_now += 1
                ratio = completed_now / max(1, total_rows)
                percent = start_percent + round(
                    min(1.0, ratio) * (end_percent - start_percent)
                )
                detail = (
                    "Собираем независимые ответы с актуальным веб-поиском"
                    if mode == "web"
                    else "Проверяем, что модели знают без веб-поиска"
                )
                await update_progress(
                    run_id,
                    stage="web_visibility" if mode == "web" else "knowledge_gap",
                    percent=percent,
                    detail=f"{detail}: {round(min(1.0, ratio) * 100)}%.",
                    eta_seconds=max(
                        300,
                        int((900 if mode == "web" else 600) * (1 - min(1.0, ratio))),
                    ),
                )

    await asyncio.gather(*(worker(job) for job in jobs))
    async with SessionLocal() as session:
        statuses = list(
            (
                await session.execute(
                    select(ModelAnswer.status).where(
                        ModelAnswer.run_id == run_id,
                        ModelAnswer.mode == mode,
                    )
                )
            ).scalars()
        )
    if not statuses or statuses.count("completed") / len(statuses) < 0.60:
        raise OpenRouterError(f"Too few successful {mode} panel responses")


async def _answers_for_catalog(run_id: str) -> list[dict[str, Any]]:
    async with SessionLocal() as session:
        rows = (
            await session.execute(
                select(ModelAnswer, VisibilityPrompt)
                .join(VisibilityPrompt, ModelAnswer.prompt_id == VisibilityPrompt.id)
                .where(
                    ModelAnswer.run_id == run_id,
                    ModelAnswer.status == "completed",
                )
                .order_by(
                    VisibilityPrompt.sequence,
                    ModelAnswer.mode,
                    ModelAnswer.provider_key,
                )
            )
        ).all()
    labels = {model.key: model.label for model in panel_models()}
    return [
        {
            "answer_id": answer.id,
            "scenario": prompt.text,
            "scenario_role": prompt.role,
            "system": labels.get(answer.provider_key, answer.provider_key),
            "mode": answer.mode,
            "answer": (answer.response_text or "")[:ANSWER_ANALYSIS_CHAR_LIMIT],
        }
        for answer, prompt in rows
    ]


def _volume_bounded_chunks(
    items: list[dict[str, Any]],
    *,
    text_key: str,
    max_items: int,
    max_chars: int,
) -> list[list[dict[str, Any]]]:
    """Keep LLM batches bounded without truncating every long answer too early."""

    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_chars = 0
    for item in items:
        item_chars = len(str(item.get(text_key) or ""))
        if current and (
            len(current) >= max_items
            or current_chars + item_chars > max_chars
        ):
            chunks.append(current)
            current = []
            current_chars = 0
        current.append(item)
        current_chars += item_chars
    if current:
        chunks.append(current)
    return chunks


async def _entity_catalog(
    run_id: str,
    profile: dict[str, Any],
    answers: list[dict[str, Any]],
) -> dict[str, Any]:
    target = {
        "brand_name": profile.get("brand_name"),
        "aliases": profile.get("brand_aliases") or [],
        "products": profile.get("products") or [],
        "entity_scope": profile.get("entity_scope") or [],
    }
    merge_target = {
        **target,
        "topics": profile.get("topics") or [],
    }
    extraction_system = f"""
Извлеки полный каталог собственных имён из небольшого пакета ответов:
бренды, компании, коммерческие продукты и платформы. Не выбирай только
«главные» сущности: сохрани каждое явно названное решение, если оно может
участвовать в выборе пользователя. Объедини орфографические варианты и алиасы.

Основной бренд отметь target/exact_target. Его подтверждённые продукты,
дочерние бренды и бизнес-юниты отметь target/portfolio_entity, но только если
связь следует из профиля или текста ответа. Альтернативы в той же задаче выбора
отметь competitor/competitor. Источники, медиа, рейтинги, рекламные кабинеты и
прочие сущности отметь other, а не конкурентами. Не приписывай цель по сходству
названия.

Для каждой сущности задай mention_policy. standalone подходит основному бренду,
конкуренту или отличимому собственному имени продукта/платформы: например,
Atlas One, Northline Hub или Orion Suite. requires_target_attribution обязателен
для общих категорий, услуг и возможностей: доставка, аналитика, консалтинг,
страхование, креатив или автоматизация. Само наличие такого слова в ответе не
означает, что модель назвала продукт целевого бренда: рядом должно быть явное
имя цели.
Не превращай общую тему пользовательского вопроса в продукт цели.
Обычные строковые aliases наследуют mention_policy сущности. Если у отличимого
имени есть общий короткий алиас с другой политикой, верни такой alias объектом
{{"value": "...", "match_policy": "requires_target_attribution"}}. Например,
Orion Suite может быть standalone, но общий alias suite требует атрибуции.

В evidence коротко укажи, откуда следует классификация.
Не вычисляй метрики.

{LIVE_RUSSIAN_RULES}
""".strip()
    chunk_jobs: list[dict[str, Any]] = []
    chunks = _volume_bounded_chunks(
        answers,
        text_key="answer",
        max_items=8,
        max_chars=ENTITY_CATALOG_CHUNK_CHAR_LIMIT,
    )
    for chunk_index, chunk in enumerate(chunks, start=1):
        digest = hashlib.sha1(
            json.dumps(
                [
                    {
                        "answer_id": item.get("answer_id"),
                        "answer": item.get("answer"),
                    }
                    for item in chunk
                ],
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:12]
        chunk_jobs.append(
            {
                "chunk_index": chunk_index,
                "artifact_key": f"entity_catalog_chunk_{chunk_index}_{digest}",
                "schema_name": f"aiv_entity_catalog_chunk_{chunk_index}",
                "answers": chunk,
            }
        )

    semaphore = asyncio.Semaphore(PROCESSING_BATCH_CONCURRENCY)

    async def extract_chunk(job: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            return await _processing_artifact(
                run_id,
                stage_key="knowledge_gap",
                artifact_key=str(job["artifact_key"]),
                schema=ENTITY_CATALOG_SCHEMA,
                schema_name=str(job["schema_name"]),
                system=extraction_system,
                user_payload={
                    "target": target,
                    "answers": job["answers"],
                },
                max_tokens=10_000,
                prompt_version=ENTITY_CATALOG_CHUNK_VERSION,
            )

    # asyncio.gather returns values in input order even when later chunks
    # finish first. The merge input and cache keys therefore stay stable.
    chunk_outcomes = await asyncio.gather(
        *(extract_chunk(job) for job in chunk_jobs),
        return_exceptions=True,
    )
    chunk_catalogs: list[dict[str, Any]] = []
    for outcome in chunk_outcomes:
        if isinstance(outcome, BaseException):
            raise outcome
        chunk_catalogs.append(outcome)

    merge_system = f"""
Собери единый исчерпывающий каталог сущностей из результатов пакетного
извлечения. Объедини только реальные алиасы и орфографические варианты одной
сущности. Не объединяй материнский бренд с отдельным продуктом или бизнес-юнитом.
Не удаляй именованную сущность только потому, что она встретилась один раз.

Основной бренд — target/exact_target. Подтверждённые продукты, дочерние бренды
и бизнес-юниты цели — target/portfolio_entity. Альтернативы в той же задаче
выбора — competitor/competitor. Источники, медиа, рейтинги и общие технологии
— other. Если пакетные классификации конфликтуют, используй профиль и evidence,
а сомнение сохрани в uncertainties.

Сохрани distinction mention_policy: отличимое собственное имя продукта или
платформы может быть standalone; общая услуга, категория или возможность
всегда requires_target_attribution. Не повышай общие слова DOOH, programmatic,
аналитика, креатив, контекстная реклама и похожие категории до standalone
только потому, что целевой бренд оказывает такую услугу. Верни полный каталог,
не метрики. Сохрани alias-level match_policy для смешанных случаев: строковый
alias наследует mention_policy сущности, объект alias задаёт явное исключение.

Каждый canonical_name верни ровно один раз. Если одна и та же именованная
сущность в пакетах попала и в target, и в competitor/other, разреши конфликт
по профилю сайта: подтверждённый продукт или бизнес-юнит цели остаётся
target/portfolio_entity, а конфликтующую дублирующую запись не возвращай.

Для подтверждённых услуг цели сверь aliases с target.products и target.topics.
Сохрани реально встреченные языковые, транслитерированные и орфографические
варианты одной услуги: например, английское performance и русское «перформанс».
Если полная форма склоняется, добавь устойчивую базовую форму, которая
буквально встречается в исходном материале. Все такие общие варианты верни
только как alias-объекты с match_policy=requires_target_attribution. Это
помогает не терять явную связь с брендом, но не разрешает считать тему запроса
его продуктом.

{LIVE_RUSSIAN_RULES}
""".strip()
    merged = await _processing_artifact(
        run_id,
        stage_key="knowledge_gap",
        artifact_key="entity_catalog",
        schema=ENTITY_CATALOG_SCHEMA,
        schema_name="aiv_entity_catalog",
        system=merge_system,
        user_payload={
            "target": merge_target,
            "chunk_catalogs": chunk_catalogs,
        },
        max_tokens=24_000,
        prompt_version=ENTITY_CATALOG_VERSION,
    )
    return _scope_entity_catalog_to_profile(merged, profile)


def _annotation_matches_answer(
    annotation: dict[str, Any],
    *,
    answer_text: str,
    answer_model: str,
    annotation_input_sha256: str,
) -> bool:
    """Accept an annotation only when every provenance field is current."""

    return bool(
        answer_text.strip()
        and annotation.get("_annotation_version") == ANNOTATION_VERSION
        and annotation.get("_answer_sha256")
        == hashlib.sha256(answer_text.encode("utf-8")).hexdigest()
        and annotation.get("_answer_model") == answer_model
        and annotation.get("_annotation_input_sha256")
        == annotation_input_sha256
    )


async def _unannotated_answers(
    run_id: str,
    *,
    annotation_input_sha256: str,
) -> list[dict[str, Any]]:
    async with SessionLocal() as session:
        rows = (
            await session.execute(
                select(ModelAnswer, VisibilityPrompt, AnswerAnnotation)
                .join(VisibilityPrompt, ModelAnswer.prompt_id == VisibilityPrompt.id)
                .outerjoin(
                    AnswerAnnotation,
                    AnswerAnnotation.answer_id == ModelAnswer.id,
                )
                .where(
                    ModelAnswer.run_id == run_id,
                    ModelAnswer.status == "completed",
                )
                .order_by(ModelAnswer.id)
            )
        ).all()
    labels = {model.key: model.label for model in panel_models()}
    output: list[dict[str, Any]] = []
    for answer, prompt, annotation in rows:
        answer_text = answer.response_text or ""
        if not answer_text.strip():
            continue
        answer_sha256 = hashlib.sha256(answer_text.encode("utf-8")).hexdigest()
        stored = annotation.annotation_json if annotation is not None else {}
        if _annotation_matches_answer(
            stored,
            answer_text=answer_text,
            answer_model=answer.model,
            annotation_input_sha256=annotation_input_sha256,
        ):
            continue
        output.append({
            "answer_id": answer.id,
            "mode": answer.mode,
            "system": labels.get(answer.provider_key, answer.provider_key),
            "answer_model": answer.model,
            "answer_sha256": answer_sha256,
            "scenario": prompt.text,
            "scenario_role": prompt.role,
            "intent_class": prompt.intent_class,
            "answer": answer_text[:ANSWER_ANALYSIS_CHAR_LIMIT],
        })
    return output


async def _save_annotations(
    run_id: str,
    annotations: list[dict[str, Any]],
    allowed_ids: set[int],
) -> int:
    saved = 0
    async with SessionLocal() as session:
        for item in annotations:
            answer_id = item.get("answer_id")
            if not isinstance(answer_id, int) or answer_id not in allowed_ids:
                continue
            existing = (
                await session.execute(
                    select(AnswerAnnotation).where(
                        AnswerAnnotation.answer_id == answer_id
                    )
                )
            ).scalar_one_or_none()
            if existing is None:
                session.add(
                    AnswerAnnotation(
                        answer_id=answer_id,
                        annotation_json=item,
                    )
                )
            else:
                existing.annotation_json = item
                flag_modified(existing, "annotation_json")
            saved += 1
        await session.commit()
    return saved


def _alias_is_present(answer_text: str, alias: str) -> bool:
    return bool(_alias_spans(answer_text, [alias]))


_PORTFOLIO_MATCH_POLICIES = frozenset(
    {"standalone", "requires_target_attribution"}
)
_PROFILE_PORTFOLIO_RELATIONSHIPS = frozenset(
    {"owned_by", "operated_by", "offered_by"}
)


def _catalog_alias_value(value: Any) -> str:
    if isinstance(value, dict):
        return str(
            value.get("value")
            or value.get("alias")
            or value.get("text")
            or ""
        ).strip()
    return str(value or "").strip()


def _entity_alias_values(entity: dict[str, Any]) -> list[str]:
    values = [
        str(entity.get("canonical_name") or "").strip(),
        *[
            _catalog_alias_value(value)
            for value in entity.get("aliases") or []
        ],
    ]
    return list(
        dict.fromkeys(value for value in values if len(value) >= 3)
    )


def _portfolio_mention_policy(
    entity: dict[str, Any],
    _profile: dict[str, Any] | None = None,
) -> str:
    """Use the declared policy; unknown legacy entries fail closed."""

    declared = str(entity.get("mention_policy") or "").casefold()
    if declared in _PORTFOLIO_MATCH_POLICIES:
        return declared
    return "requires_target_attribution"


_CYRILLIC_TO_LATIN = str.maketrans(
    {
        "а": "a",
        "б": "b",
        "в": "v",
        "г": "g",
        "д": "d",
        "е": "e",
        "ё": "e",
        "ж": "zh",
        "з": "z",
        "и": "i",
        "й": "i",
        "к": "k",
        "л": "l",
        "м": "m",
        "н": "n",
        "о": "o",
        "п": "p",
        "р": "r",
        "с": "s",
        "т": "t",
        "у": "u",
        "ф": "f",
        "х": "kh",
        "ц": "ts",
        "ч": "ch",
        "ш": "sh",
        "щ": "shch",
        "ъ": "",
        "ы": "y",
        "ь": "",
        "э": "e",
        "ю": "yu",
        "я": "ya",
    }
)


def _loanword_key(value: str) -> str:
    """Normalize a narrow Latin/Cyrillic loanword equivalence.

    This is deliberately not fuzzy matching.  It only supports deterministic
    transliteration plus the common terminal ``-ce`` → ``-с`` spelling, as in
    ``performance`` / ``перформанс``.
    """

    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    normalized = normalized.translate(_CYRILLIC_TO_LATIN)
    normalized = re.sub(r"[^a-z0-9]+", "", normalized)
    normalized = re.sub(r"ce$", "s", normalized)
    return normalized


def _bounded_edit_distance(left: str, right: str, limit: int) -> int:
    """Return an edit distance capped above ``limit`` for short aliases."""

    if abs(len(left) - len(right)) > limit:
        return limit + 1
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        row_minimum = left_index
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1]
                    + (left_char != right_char),
                )
            )
            row_minimum = min(row_minimum, current[-1])
        if row_minimum > limit:
            return limit + 1
        previous = current
    return previous[-1]


def _is_near_target_alias(value: str, target_aliases: list[str]) -> bool:
    """Recognize a close transliterated spelling, not an arbitrary owner."""

    value_tokens = re.findall(r"[a-zа-я0-9]+", value.casefold(), re.UNICODE)
    if len(value_tokens) != 1:
        return False
    value_key = _loanword_key(value_tokens[0])
    if len(value_key) < 6:
        return False
    for alias in target_aliases:
        alias_tokens = re.findall(
            r"[a-zа-я0-9]+",
            str(alias).casefold(),
            re.UNICODE,
        )
        for alias_token in alias_tokens:
            alias_key = _loanword_key(alias_token)
            if (
                len(alias_key) >= 6
                and _bounded_edit_distance(value_key, alias_key, 2) <= 2
            ):
                return True
    return False


def _profile_name_confirms_contextual_alias(
    trusted_name: str,
    alias: str,
) -> bool:
    """Confirm a generic alias from the site's own entity vocabulary.

    The answer-derived catalog may suggest helper aliases, but it cannot make
    an unrelated word a client offering.  Exact/inflected matches remain the
    default.  Two conservative equivalences cover ordinary site language:
    Russian noun→adjective service labels (``креатив`` / ``креативные``) and
    exact cross-script loanwords (``performance`` / ``перформанс``).
    """

    if _contextual_alias_is_present(trusted_name, alias):
        return True
    normalized_alias = unicodedata.normalize("NFKC", alias).casefold().strip()
    alias_tokens = re.findall(r"[a-zа-я0-9]+", normalized_alias, re.UNICODE)
    if len(alias_tokens) != 1 or len(alias_tokens[0]) < 5:
        return False
    alias_token = alias_tokens[0]
    trusted_tokens = re.findall(
        r"[a-zа-я0-9]+",
        unicodedata.normalize("NFKC", trusted_name).casefold(),
        re.UNICODE,
    )
    russian_adjective_suffixes = {
        "ная",
        "ного",
        "ное",
        "ной",
        "ному",
        "ную",
        "ные",
        "ный",
        "ным",
        "ными",
        "ных",
    }
    for trusted_token in trusted_tokens:
        if (
            re.fullmatch(r"[а-я]+", alias_token, re.UNICODE)
            and re.fullmatch(r"[а-я]+", trusted_token, re.UNICODE)
            and trusted_token.startswith(alias_token)
            and trusted_token[len(alias_token) :] in russian_adjective_suffixes
        ):
            return True
        cross_script = bool(
            re.search(r"[a-z]", alias_token)
            and re.search(r"[а-я]", trusted_token)
        ) or bool(
            re.search(r"[а-я]", alias_token)
            and re.search(r"[a-z]", trusted_token)
        )
        if not cross_script:
            continue
        alias_key = _loanword_key(alias_token)
        trusted_key = _loanword_key(trusted_token)
        if len(alias_key) >= 6 and alias_key == trusted_key:
            return True
    return False


def _entity_alias_entries(
    entity: dict[str, Any],
    profile: dict[str, Any] | None = None,
    *,
    excluded_aliases: list[str] | None = None,
) -> list[tuple[str, str]]:
    """Return fail-closed aliases with their deterministic match policy."""

    default_policy = _portfolio_mention_policy(entity, profile)
    canonical_name = str(entity.get("canonical_name") or "").strip()
    candidates: list[tuple[str, str, bool]] = [
        (canonical_name, default_policy, True)
    ]
    for raw_alias in entity.get("aliases") or []:
        alias = _catalog_alias_value(raw_alias)
        policy = (
            str(raw_alias.get("match_policy") or "").casefold()
            if isinstance(raw_alias, dict)
            else default_policy
        )
        if policy not in _PORTFOLIO_MATCH_POLICIES:
            policy = default_policy
        candidates.append((alias, policy, False))

    trusted_names = [
        str(value or "").strip()
        for value in entity.get("_profile_confirmed_match_aliases") or []
        if str(value or "").strip()
    ]
    if profile is not None and _catalog_marks_portfolio_entity(entity):
        trusted_names = _profile_confirmed_names_for_entity(entity, profile)
    trusted_normalized = {
        value.casefold().replace("ё", "е").strip()
        for value in trusted_names
    }
    provenance_guarded = bool(
        _catalog_marks_portfolio_entity(entity)
        and (
            profile is not None
            or "_profile_confirmed_match_aliases" in entity
        )
    )

    excluded_normalized = {
        value.casefold().replace("ё", "е").strip()
        for value in excluded_aliases or []
        if len(value.strip()) >= 3
    }
    entries: dict[str, tuple[str, str]] = {}
    for alias, policy, is_canonical in candidates:
        normalized = alias.casefold().replace("ё", "е").strip()
        if len(normalized) < 3 or normalized in excluded_normalized:
            continue
        lexical_tokens = re.findall(r"[a-zа-я0-9]+", normalized, re.UNICODE)
        profile_confirms_contextual_alias = bool(
            policy == "requires_target_attribution"
            and any(
                _profile_name_confirms_contextual_alias(
                    trusted_name,
                    alias,
                )
                for trusted_name in trusted_names
            )
        )
        if (
            provenance_guarded
            and not is_canonical
            and normalized not in trusted_normalized
            and len(lexical_tokens) <= 1
            and not profile_confirms_contextual_alias
        ):
            # A catalog-produced one-word helper such as ``Group``, ``онлайн``
            # or ``покупка`` is too weak to identify a portfolio entity.
            # Even target attribution cannot make an answer-derived alias a
            # real client offering. A contextual one-word alias survives only
            # when the site's own profile independently confirms that alias
            # for this entity (for example ``DOOH`` or ``аналитика``).
            continue
        if (
            len(lexical_tokens) == 1
            and any(
                normalized
                in re.findall(r"[a-zа-я0-9]+", excluded, re.UNICODE)
                for excluded in excluded_normalized
            )
        ):
            continue
        if (
            provenance_guarded
            and policy == "standalone"
            and normalized not in trusted_normalized
            and not is_canonical
        ):
            # Answer-derived aliases cannot inherit standalone status from a
            # canonical product.  Only exact site-profile names may do so.
            policy = "requires_target_attribution"
        existing = entries.get(normalized)
        if (
            existing is None
            or policy == "requires_target_attribution"
        ):
            # Conflicting duplicate declarations resolve to the safer policy.
            entries[normalized] = (alias, policy)
    return list(entries.values())


def _target_aliases(
    profile: dict[str, Any],
    _catalog: dict[str, Any],
) -> list[str]:
    """Return only aliases grounded in the analyzed site's own profile.

    The answer-derived entity catalog is deliberately not authoritative for
    the identity of the target.  Otherwise a broad alias invented while
    reading panel answers can turn an ordinary word into a brand mention.
    """

    aliases = [
        str(profile.get("brand_name") or ""),
        *[str(value) for value in profile.get("brand_aliases") or []],
    ]
    for entity in profile.get("entity_scope") or []:
        if not isinstance(entity, dict):
            continue
        relationship = str(
            entity.get("relationship") or ""
        ).casefold()
        if (
            relationship != "self"
            or str(entity.get("entity_type") or "").casefold()
            != "primary_brand"
            or entity.get("commercially_relevant") is not True
            or str(entity.get("confidence") or "").casefold() == "low"
        ):
            continue
        aliases.extend(
            [
                str(entity.get("canonical_name") or ""),
                *[
                    str(value or "")
                    for value in entity.get("aliases") or []
                ],
            ]
        )
    return list(
        dict.fromkeys(
            alias.strip()
            for alias in aliases
            if len(alias.strip()) >= 3
        )
    )


def _attribution_owner_aliases(
    profile: dict[str, Any],
    catalog: dict[str, Any],
) -> list[str]:
    """Return site-proven owners that may scope a portfolio claim in raw text.

    A project site can legitimately describe an offering through its confirmed
    developer or operator.  Only high/medium-confidence primary brands and
    business units with an ownership/operation relationship are allowed to
    extend the exact target aliases.  Products and services never become
    owners merely because they occur in the profile.
    """

    aliases = list(_target_aliases(profile, catalog))
    for scoped in profile.get("entity_scope") or []:
        if not isinstance(scoped, dict):
            continue
        relationship = str(scoped.get("relationship") or "").casefold()
        entity_type = str(scoped.get("entity_type") or "").casefold()
        confidence = str(scoped.get("confidence") or "").casefold()
        if (
            relationship not in {"self", "owned_by", "operated_by"}
            or entity_type not in {"primary_brand", "business_unit"}
            or scoped.get("commercially_relevant") is not True
            or confidence == "low"
        ):
            continue
        aliases.extend(
            [
                str(scoped.get("canonical_name") or ""),
                *[
                    str(value or "")
                    for value in scoped.get("aliases") or []
                ],
            ]
        )
    return list(
        dict.fromkeys(
            alias.strip()
            for alias in aliases
            if len(alias.strip()) >= 3
        )
    )


def _entity_attribution_aliases(
    profile: dict[str, Any],
    catalog: dict[str, Any],
    entity: dict[str, Any],
) -> list[str]:
    """Return attribution actors permitted for one portfolio entity.

    Direct target aliases are always valid actors.  An upstream owner such as
    a developer may scope a generic claim only when that owner is part of the
    entity's declared identity (for example, ``Онлайн-покупка MR Group``).
    This prevents one group company from donating every feature of sibling
    products to the site being analyzed.
    """

    direct = _target_aliases(profile, catalog)
    direct_keys = {
        value.casefold().replace("ё", "е").strip() for value in direct
    }
    # The answer-derived alias list is not authoritative for ownership.  A
    # catalog model could otherwise attach ``MR Group`` to any confirmed JOIS
    # feature and make a sibling project's claim look target-owned.  Only the
    # names independently confirmed by the site profile may opt an upstream
    # owner into this entity's attribution scope.  The answer-derived
    # canonical is deliberately excluded too: otherwise a model can append an
    # owner name to an otherwise matching canonical and reopen sibling
    # donation through that suffix.
    confirmed_names = entity.get("_profile_confirmed_match_aliases")
    if not isinstance(confirmed_names, list):
        confirmed_names = _profile_confirmed_names_for_entity(entity, profile)
    identity = [
        str(value or "").strip()
        for value in confirmed_names
        if str(value or "").strip()
    ]
    allowed = list(direct)
    for owner in _attribution_owner_aliases(profile, catalog):
        owner_key = owner.casefold().replace("ё", "е").strip()
        if owner_key in direct_keys:
            continue
        if any(_alias_is_present(value, owner) for value in identity):
            allowed.append(owner)
    return list(dict.fromkeys(allowed))


def _entity_attribution_alias_map(
    profile: dict[str, Any],
    catalog: dict[str, Any],
) -> dict[str, list[str]]:
    return {
        str(entity.get("canonical_name") or "").casefold(): (
            _entity_attribution_aliases(profile, catalog, entity)
        )
        for entity in catalog.get("entities") or []
        if isinstance(entity, dict) and entity.get("canonical_name")
    }


def _catalog_marks_portfolio_entity(entity: dict[str, Any]) -> bool:
    if (
        entity.get("category") != "target"
        or entity.get("commercially_relevant", True) is not True
    ):
        return False
    relationship = str(
        entity.get("target_relationship")
        or entity.get("relationship")
        or ""
    ).strip().casefold()
    return relationship in {
        "portfolio_entity",
        *_PROFILE_PORTFOLIO_RELATIONSHIPS,
    }


def _profile_confirmed_names_for_entity(
    entity: dict[str, Any],
    profile: dict[str, Any],
) -> list[str]:
    """Return the exact site-profile names that confirm one canonical entity.

    Matching starts from the catalog canonical name only.  Catalog aliases are
    intentionally excluded because they are produced from panel answers and
    cannot serve as proof of portfolio membership.
    """

    canonical = str(entity.get("canonical_name") or "").strip()
    if not canonical:
        return []

    def identity_text(value: str) -> str:
        return (
            unicodedata.normalize("NFKC", value)
            .casefold()
            .replace("ё", "е")
            .strip()
        )

    def identity_tokens(value: str) -> list[str]:
        return [
            token
            for token in re.findall(
                r"[^\W_]+",
                identity_text(value),
                re.UNICODE,
            )
            if token not in ignored_tokens
        ]

    # Ignore only connective/site-type noise.  Owner and brand tokens are
    # identity-bearing on the untrusted catalog side: dropping ``Group`` or a
    # target-brand suffix here would let an answer-derived canonical append an
    # invented owner and still inherit membership from a shorter profile name.
    # Exact names still pass above, while a concise catalog name may remain a
    # leading prefix of a more descriptive trusted profile name.
    ignored_tokens = {
        "жк",
        "для",
        "and",
        "the",
        "of",
        "a",
        "и",
        "от",
        "у",
        "в",
        "на",
        "по",
        "с",
    }
    for brand_value in (
        str(profile.get("brand_name") or ""),
        *[str(value or "") for value in profile.get("brand_aliases") or []],
    ):
        ignored_tokens.update(
            re.findall(r"[^\W_]+", identity_text(brand_value), re.UNICODE)
        )

    def canonical_matches(profile_name: str) -> bool:
        normalized_canonical = identity_text(canonical)
        normalized_profile = identity_text(profile_name)
        if normalized_canonical == normalized_profile:
            return True
        canonical_tokens = identity_tokens(canonical)
        profile_tokens = identity_tokens(profile_name)
        # Membership is asymmetric: the trusted profile may be more verbose
        # than its concise base name, but that base must be the leading identity
        # of the trusted name.  A generic tail such as ``Enterprise Analytics``
        # cannot inherit membership from ``Campaign 360 Enterprise Analytics``;
        # an invented suffix cannot inherit it in the opposite direction.
        return bool(
            len(canonical_tokens) >= 2
            and profile_tokens[: len(canonical_tokens)] == canonical_tokens
        )

    confirmed: list[str] = []
    for scoped in profile.get("entity_scope") or []:
        if not isinstance(scoped, dict):
            continue
        relationship = str(scoped.get("relationship") or "").casefold()
        if (
            relationship not in _PROFILE_PORTFOLIO_RELATIONSHIPS
            or scoped.get("commercially_relevant") is not True
            or str(scoped.get("confidence") or "").casefold() == "low"
        ):
            continue
        scoped_names = [
            str(scoped.get("canonical_name") or "").strip(),
            *[
                str(value or "").strip()
                for value in scoped.get("aliases") or []
            ],
        ]
        if any(canonical_matches(value) for value in scoped_names if value):
            confirmed.extend(scoped_names)

    for product in profile.get("products") or []:
        product_name = str(product or "").strip()
        if product_name and canonical_matches(product_name):
            confirmed.append(product_name)
    return list(dict.fromkeys(value for value in confirmed if value))


def _profile_confirms_portfolio_entity(
    entity: dict[str, Any],
    profile: dict[str, Any],
) -> bool:
    """Accept portfolio membership only from the analyzed site profile."""

    if not _catalog_marks_portfolio_entity(entity):
        return False
    return bool(_profile_confirmed_names_for_entity(entity, profile))


def _scoped_entity_resolution_priority(entity: dict[str, Any]) -> int:
    """Rank same-name records after membership has been checked.

    The catalog is produced by merging independent LLM batches. A named
    portfolio entity can therefore occasionally survive twice, for example
    once as the target's product and once as a competitor. Downstream maps are
    keyed by ``canonical_name``; resolving that collision here prevents list
    order from silently changing the entity's scope.
    """

    relationship = str(
        entity.get("target_relationship")
        or entity.get("relationship")
        or ""
    ).casefold()
    if (
        entity.get("category") == "target"
        and relationship in {"exact_target", "self"}
    ):
        return 40
    if (
        _catalog_marks_portfolio_entity(entity)
        and entity.get("_profile_membership_confirmed") is True
    ):
        return 30
    if (
        entity.get("category") == "competitor"
        or relationship == "competitor"
    ):
        return 20
    if entity.get("category") == "other" or relationship == "unrelated":
        return 10
    return 0


def _deduplicate_scoped_catalog_entities(
    entities: list[Any],
) -> list[dict[str, Any]]:
    """Keep one deterministic record per canonical name, fail-closed.

    A profile-confirmed target wins over an answer-derived competitor with the
    same name. An unconfirmed target candidate is demoted before this helper is
    called, so it cannot displace a real competitor merely because an answer
    claimed ownership.
    """

    resolved: list[dict[str, Any]] = []
    positions: dict[str, int] = {}
    for raw_entity in entities:
        if not isinstance(raw_entity, dict):
            continue
        entity = raw_entity
        canonical = str(entity.get("canonical_name") or "").strip()
        if not canonical:
            continue
        key = canonical.casefold().replace("ё", "е")
        position = positions.get(key)
        if position is None:
            positions[key] = len(resolved)
            resolved.append(entity)
            continue
        if _scoped_entity_resolution_priority(
            entity
        ) > _scoped_entity_resolution_priority(resolved[position]):
            resolved[position] = entity
    return resolved


def _scope_entity_catalog_to_profile(
    catalog: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Prevent answer-derived claims from expanding the target portfolio."""

    scoped_catalog = copy.deepcopy(catalog)
    for entity in scoped_catalog.get("entities") or []:
        if not isinstance(entity, dict):
            continue
        if not _catalog_marks_portfolio_entity(entity):
            continue
        confirmed_names = _profile_confirmed_names_for_entity(entity, profile)
        confirmed = bool(confirmed_names)
        entity["_profile_membership_confirmed"] = confirmed
        entity["_profile_confirmed_match_aliases"] = confirmed_names
        if confirmed:
            continue
        entity["category"] = "other"
        entity["target_relationship"] = "unrelated"
        entity["commercially_relevant"] = False
        entity["_scope_rejection_reason"] = (
            "Ответ модели не может расширять портфель без подтверждения "
            "в профиле исследуемого сайта."
        )
    scoped_catalog["entities"] = _deduplicate_scoped_catalog_entities(
        list(scoped_catalog.get("entities") or [])
    )
    return scoped_catalog


def _alias_spans(answer_text: str, aliases: list[str]) -> list[tuple[int, int]]:
    normalized_text = answer_text.casefold().replace("ё", "е")
    spans: list[tuple[int, int]] = []
    for alias in aliases:
        normalized_alias = alias.casefold().replace("ё", "е").strip()
        if len(normalized_alias) < 3:
            continue
        for match in re.finditer(
            rf"(?<![\w]){re.escape(normalized_alias)}(?![\w])",
            normalized_text,
            re.UNICODE,
        ):
            start, end = match.span()
            token_start = start
            while (
                token_start > 0
                and normalized_text[token_start - 1]
                not in " \t\r\n<>[](){}\"'«»“”"
            ):
                token_start -= 1
            token_end = end
            while (
                token_end < len(normalized_text)
                and normalized_text[token_end]
                not in " \t\r\n<>[](){}\"'«»“”"
            ):
                token_end += 1
            token = normalized_text[token_start:token_end].strip(".,;:")
            if (
                re.match(r"^(?:https?://|www\.|mailto:)", token)
                or "@" in token
                or re.match(
                    r"^(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/|$)",
                    token,
                )
            ):
                continue
            spans.append((start, end))
    return spans


_RUSSIAN_INFLECTION_ENDINGS = tuple(
    sorted(
        {
            "иями",
            "ями",
            "ами",
            "ием",
            "ией",
            "ого",
            "ему",
            "ими",
            "ыми",
            "ение",
            "ения",
            "ению",
            "ении",
            "ия",
            "ии",
            "ию",
            "ий",
            "ая",
            "яя",
            "ое",
            "ее",
            "ые",
            "ие",
            "ов",
            "ев",
            "ом",
            "ем",
            "ам",
            "ям",
            "ах",
            "ях",
            "ой",
            "ей",
            "а",
            "я",
            "у",
            "ю",
            "е",
            "и",
            "ы",
            "о",
        },
        key=len,
        reverse=True,
    )
)
_RUSSIAN_INFLECTION_PATTERN = (
    "(?:" + "|".join(map(re.escape, _RUSSIAN_INFLECTION_ENDINGS)) + ")?"
)


def _contextual_alias_spans(
    answer_text: str,
    aliases: list[str],
) -> list[tuple[int, int]]:
    """Match conservative Russian inflections for attribution-only aliases.

    Each Russian token remains in place and may take a conservative ending.
    This covers ``зелёные террасы`` → ``зелёными террасами`` and
    ``онлайн-покупка`` → ``онлайн-покупки`` without turning component words
    into independent aliases.
    """

    spans = set(_alias_spans(answer_text, aliases))
    normalized_text = answer_text.casefold().replace("ё", "е")
    for alias in aliases:
        normalized_alias = alias.casefold().replace("ё", "е").strip()
        token_matches = list(re.finditer(r"[а-я]+", normalized_alias))
        if not token_matches:
            continue
        pattern_parts: list[str] = []
        cursor = 0
        inflected = False
        for token_match in token_matches:
            pattern_parts.append(
                re.escape(normalized_alias[cursor : token_match.start()])
            )
            token = token_match.group(0)
            base = token
            if len(token) >= 5:
                for ending in _RUSSIAN_INFLECTION_ENDINGS:
                    if base.endswith(ending) and len(base) - len(ending) >= 5:
                        base = base[: -len(ending)]
                        break
                pattern_parts.append(
                    re.escape(base) + _RUSSIAN_INFLECTION_PATTERN
                )
                inflected = True
            else:
                pattern_parts.append(re.escape(token))
            cursor = token_match.end()
        pattern_parts.append(re.escape(normalized_alias[cursor:]))
        if not inflected:
            continue
        alias_pattern = "".join(pattern_parts)
        spans.update(
            match.span()
            for match in re.finditer(
                rf"(?<![\w]){alias_pattern}(?![\w])",
                normalized_text,
                re.UNICODE,
            )
        )
    return sorted(spans)


def _contextual_alias_is_present(answer_text: str, alias: str) -> bool:
    return bool(_contextual_alias_spans(answer_text, [alias]))


def _normalized_evidence_text(value: str) -> str:
    normalized = value.casefold().replace("ё", "е")
    normalized = re.sub(r"[*_`#]+", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip(" \t\r\n«»\"'“”„")


def _evidence_is_literal(answer_text: str, evidence: str) -> bool:
    """Require the persisted quote to be an exact contiguous raw substring."""

    candidate = evidence.strip()
    return len(candidate) >= 3 and candidate in answer_text


def _literal_alias_evidence(
    answer_text: str,
    alias_entries: list[tuple[str, str]],
) -> str:
    """Return the shortest exact raw slice matching one complete alias."""

    candidates: set[str] = set()
    for alias, policy in alias_entries:
        spans = (
            _alias_spans(answer_text, [alias])
            if policy == "standalone"
            else _contextual_alias_spans(answer_text, [alias])
        )
        for start, end in spans:
            candidate = answer_text[start:end]
            if candidate and candidate in answer_text:
                candidates.add(candidate)
    return min(candidates, key=lambda value: (len(value), value)) if candidates else ""


def _evidence_contains_complete_alias(
    evidence: str,
    alias_entries: list[tuple[str, str]],
) -> bool:
    """Reject truncated brand/product tokens even when they are substrings."""

    return any(
        (
            _alias_is_present(evidence, alias)
            if policy == "standalone"
            else _contextual_alias_is_present(evidence, alias)
        )
        for alias, policy in alias_entries
    )


_STRUCTURED_ATTRIBUTION_LABEL = re.compile(
    r"(?<![\w])(?:"
    r"сильная\s+сторона|"
    r"в\s+ч[её]м\s+(?:сила|сильная\s+сторона)|"
    r"специализация|"
    r"подходит\s+для|"
    r"хорошо\s+сочетает|"
    r"экспертиза|"
    r"компетенции|"
    r"аналитика\s+и\s+(?:технологии|по)|"
    r"предлага\w*"
    r")(?![\w])",
    re.UNICODE,
)
_STRUCTURED_OWNER_PREFIX = re.compile(
    r"^[\s\d.):|/\\—–\-()[\]{}>#•·]*$",
    re.UNICODE,
)
_STRUCTURED_SEPARATOR = re.compile(
    r"^[\s.:|/\\—–\-()[\]{}>#•·]*$",
    re.UNICODE,
)
_STRUCTURED_ANY_FIELD = re.compile(
    r"^[\s*_\-•·]*(?:\*\*)?[^:\n]{2,80}:(?:\*\*)?",
    re.UNICODE,
)
_STRUCTURED_FIELD_CONTINUATION = re.compile(
    r"^[\s*_\-•·]*(?:"
    r"глубок\w*|сильн\w*|хорошо|облада\w*|"
    r"уме\w*|работа\w*|подход\w*|историческ\w*|"
    r"также|кроме\s+того|дополнительно"
    r")",
    re.UNICODE,
)


def _structured_separator_only(value: str) -> bool:
    return bool(_STRUCTURED_SEPARATOR.fullmatch(value))


def _structured_label_starts_field(value: str) -> bool:
    """Tell a labelled field from the same words used as ordinary prose."""

    return any(
        _structured_separator_only(value[: match.start()])
        for match in _STRUCTURED_ATTRIBUTION_LABEL.finditer(value)
    )


def _structured_field_has_competing_owner(
    value: str,
    entity_aliases: list[str],
    target_aliases: list[str],
) -> bool:
    """Reject an explicit second actor inside a target-owned field."""

    normalized = value.casefold().replace("ё", "е")
    generic_owners = {
        "агентство",
        "агентства",
        "бренд",
        "компания",
        "команда",
        "она",
        "он",
        "они",
    }
    for entity_start, _entity_end in _contextual_alias_spans(
        normalized,
        entity_aliases,
    ):
        prefix = normalized[:entity_start]
        line_start = prefix.rfind("\n") + 1
        line_prefix = re.sub(
            r"[*_`]+",
            "",
            value[line_start:entity_start],
        )
        normalized_line_prefix = line_prefix.casefold().replace("ё", "е")
        owner_label = (
            None
            if _structured_label_starts_field(normalized_line_prefix)
            else re.match(
                r"^\s*[-*•]?\s*"
                r"([\w.-]{2,}(?:\s+[\w.-]{2,}){0,3})\s*"
                r"(?::|/|\||-|[^\w\s])(?:\s*.*)?$",
                line_prefix,
                re.UNICODE,
            )
        )
        if (
            owner_label is not None
            and not _structured_label_starts_field(line_prefix)
        ):
            label = owner_label.group(1).strip()
            normalized_label = label.casefold().replace("ё", "е")
            if (
                not _markdown_colon_label_is_field(label)
                and _STRUCTURED_ATTRIBUTION_LABEL.fullmatch(
                    normalized_label
                ) is None
                and normalized_label not in generic_owners
                and not any(
                    _alias_is_present(label, alias)
                    for alias in target_aliases
                )
            ):
                # ``- ПИК: White Box доступен`` is a competitor-owned claim,
                # even when it sits directly below a ``### JOIS`` heading.
                return True
        suffix = normalized[_entity_end : _entity_end + 180]
        postfix_owner = re.search(
            r"(?:"
            r"(?:доступ\w*|предлага\w*|предоставля\w*|"
            r"представлен\w*|оказыва\w*)[^.!?;\n]{0,96}"
            r"(?:\bот\b|\bу\b|компани(?:ей|и)|бренд(?:ом|а)|"
            r"застройщик(?:ом|а))|"
            r"(?:предлага\w*|предоставля\w*|представля\w*)\s+"
            r"(?:только|лишь|исключительно)|"
            r"(?:[—–\-,:]\s*)(?:продукт\w*|решени\w*|"
            r"сервис\w*|эксклюзив\w*|разработк\w*)|"
            r"(?:разработан\w*|создан\w*|произведен\w*)"
            r"(?:\s+компани\w*)?|"
            r"(?:предлагаем\w*|(?:с)?проектирован\w*)"
            r"(?:\s+компани\w*)?|"
            r"принадлеж\w*|от\s+компани\w*"
            r")\s+"
            r"([\w-]{2,}(?:\.[\w-]+)*"
            r"(?:\s+[\w-]{2,}(?:\.[\w-]+)*){0,2})",
            suffix,
            re.UNICODE,
        )
        if postfix_owner is not None:
            owner = postfix_owner.group(1).strip()
            if not any(
                _alias_is_present(owner, alias) for alias in target_aliases
            ):
                return True
        raw_suffix = value[_entity_end : _entity_end + 160]
        parenthetical_owner = re.match(
            r"^\s*\(\s*"
            r"([A-ZА-ЯЁ][\w.-]*(?:\s+[A-ZА-ЯЁ][\w.-]*){0,2})"
            r"\s*\)",
            raw_suffix,
            re.UNICODE,
        )
        if parenthetical_owner is not None and not any(
            _alias_is_present(parenthetical_owner.group(1), alias)
            for alias in target_aliases
        ):
            return True
        owners = list(
            re.finditer(
                r"(?<![\w])"
                r"([\w-]{2,}(?:\.[\w-]+)*"
                r"(?:\s+[\w-]{2,}(?:\.[\w-]+)*){0,3})\s+"
                r"(?:предлага\w*|оказыва\w*|"
                r"специализиру\w*|предоставля\w*|"
                r"поддержива\w*)\s*$",
                prefix,
                re.UNICODE,
            )
        )
        if not owners:
            continue
        owner = owners[-1].group(1)
        if owner in generic_owners:
            continue
        if any(_alias_is_present(owner, alias) for alias in target_aliases):
            continue
        return True
    return False


def _structured_field_mentions_entity(
    line: str,
    entity_aliases: list[str],
    *,
    target_aliases: list[str],
    continuation: str = "",
) -> bool:
    """Match an entity inside one labelled field without crossing a sentence."""

    segments = re.split(r"(?<=[.!?;])\s+", line)
    field_open = False
    for segment_index, segment in enumerate(segments):
        matched_label = False
        for label in _STRUCTURED_ATTRIBUTION_LABEL.finditer(segment):
            if not _structured_separator_only(segment[: label.start()]):
                continue
            matched_label = True
            field_open = True
            label_text = segment[label.start() : label.end()]
            if any(
                _contextual_alias_is_present(label_text, alias)
                for alias in entity_aliases
            ) and not _structured_field_has_competing_owner(
                segment,
                entity_aliases,
                target_aliases,
            ):
                return True
            field_value = re.split(
                r"[.!?;]",
                segment[label.end() :],
                maxsplit=1,
            )[0]
            if any(
                _contextual_alias_is_present(field_value, alias)
                for alias in entity_aliases
            ) and not _structured_field_has_competing_owner(
                segment,
                entity_aliases,
                target_aliases,
            ):
                return True
            if (
                segment_index != len(segments) - 1
                or not _structured_separator_only(field_value)
            ):
                continue
            continuation_value = re.split(
                r"[.!?;]",
                continuation,
                maxsplit=1,
            )[0]
            for alias in entity_aliases:
                for alias_start, _alias_end in _contextual_alias_spans(
                    continuation_value,
                    [alias],
                ):
                    if _structured_separator_only(
                        continuation_value[:alias_start]
                    ) and not _structured_field_has_competing_owner(
                        continuation_value,
                        entity_aliases,
                        target_aliases,
                    ):
                        return True
        if matched_label:
            continue
        if field_open and _STRUCTURED_FIELD_CONTINUATION.match(segment):
            if any(
                _contextual_alias_is_present(segment, alias)
                for alias in entity_aliases
            ) and not _structured_field_has_competing_owner(
                segment,
                entity_aliases,
                target_aliases,
            ):
                return True
            continue
        field_open = False
    return False


def _structured_owned_line_mentions_entity(
    line: str,
    entity_aliases: list[str],
    target_aliases: list[str],
) -> bool:
    """Accept a relational bullet under an explicit target-owned heading."""

    if _structured_field_has_competing_owner(
        line,
        entity_aliases,
        target_aliases,
    ):
        return False
    for alias_start, alias_end in _contextual_alias_spans(
        line,
        entity_aliases,
    ):
        prefix = line[:alias_start]
        suffix = line[alias_end : alias_end + 80]
        if (
            _structured_separator_only(prefix)
            and _entity_tail_has_attribution_cue(suffix)
        ) or _gap_has_attribution_cue(prefix, target_first=True):
            return True
    return False


def _structured_owner_separator_only(
    value: str,
    target_aliases: list[str],
) -> bool:
    """Allow formatting plus a repeated target alias in an owner heading."""

    remainder = value
    spans = _alias_spans(remainder, target_aliases)
    for start, end in reversed(spans):
        remainder = remainder[:start] + remainder[end:]
    remainder = re.sub(
        r"\(([^)]{2,80})\)",
        lambda match: (
            ""
            if _is_near_target_alias(match.group(1), target_aliases)
            else match.group(0)
        ),
        remainder,
    )
    remainder = re.sub(
        r"\([^)]*(?:москва|санкт[-\s]?петербург|офис|город|росси)[^)]*\)",
        "",
        remainder,
    )
    return _structured_separator_only(remainder)


def _structured_owner_prefix_only(value: str) -> bool:
    """Allow a short target-owned heading such as «У Brand:»."""

    if _STRUCTURED_OWNER_PREFIX.fullmatch(value):
        return True
    normalized = value.casefold().replace("ё", "е")
    return bool(
        re.fullmatch(
            r"[\s\d.):|/\\—–\-()[\]{}>#•·]*"
            r"(?:у|для|про|о)\s*",
            normalized,
            re.UNICODE,
        )
    )


def _has_structured_target_attribution(
    evidence: str,
    entity_aliases: list[str],
    target_aliases: list[str],
) -> bool:
    """Recognize a target-owned labelled field inside one continuous block."""

    normalized = evidence.casefold().replace("ё", "е")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[*_`]+", "", normalized)
    blocks = re.split(r"\n[ \t]*\n+", normalized)

    for block in blocks:
        lines = block.splitlines() or [block]
        for owner_index, owner_line in enumerate(lines):
            for target_alias in target_aliases:
                for target_start, target_end in _alias_spans(
                    owner_line,
                    [target_alias],
                ):
                    if not _structured_owner_prefix_only(
                        owner_line[:target_start]
                    ):
                        continue
                    owner_tail = owner_line[target_end:]
                    inline_label = _STRUCTURED_ATTRIBUTION_LABEL.search(
                        owner_tail
                    )
                    if inline_label is not None:
                        if not _structured_owner_separator_only(
                            owner_tail[: inline_label.start()],
                            target_aliases,
                        ):
                            continue
                        candidate_lines = [
                            owner_tail[inline_label.start() :],
                            *lines[owner_index + 1 :],
                        ]
                    elif _structured_owner_separator_only(
                        owner_tail,
                        target_aliases,
                    ):
                        candidate_lines = lines[owner_index + 1 :]
                    else:
                        continue

                    for candidate_index, candidate in enumerate(candidate_lines):
                        next_line = (
                            candidate_lines[candidate_index + 1]
                            if candidate_index + 1 < len(candidate_lines)
                            else ""
                        )
                        if _structured_field_mentions_entity(
                            candidate,
                            entity_aliases,
                            target_aliases=target_aliases,
                            continuation=next_line,
                        ):
                            return True
                        if _structured_owned_line_mentions_entity(
                            candidate,
                            entity_aliases,
                            target_aliases,
                        ):
                            return True
                        if (
                            _STRUCTURED_ATTRIBUTION_LABEL.search(candidate)
                            is not None
                            or _STRUCTURED_ANY_FIELD.match(candidate) is not None
                            or re.match(r"^[\s]*[-•·]", candidate) is not None
                            or _structured_separator_only(candidate)
                        ):
                            continue
                        # A free-text line or another owner starts a new fragment.
                        break
    return False


def _gap_has_attribution_cue(gap: str, *, target_first: bool) -> bool:
    compact = gap.casefold().replace("ё", "е").strip()
    if target_first:
        return bool(
            re.search(
                r"(?:"
                r"предлага\w*|оказыва\w*|предоставля\w*|развива\w*|"
                r"запуска\w*|владе\w*|управля\w*|"
                r"поддержива\w*|использу\w*|реализу\w*|"
                r"име(?:ет|ют|ла|ли|ем|ете|ть)\b|"
                r"включа\w*|охватыва\w*|"
                r"бер\w*\s+на\s+себя|"
                r"указ\w*|перечисл\w*|описыва\w*|"
                r"закуп\w*|"
                r"специализиру\w*\s+на|работа\w*\s+(?:с|в)|"
                r"сил[её]н\w*\s+в|"
                r"агентств\w*[^.!?;\n]{0,96}\b(?:с|по)\b|"
                r"хорошо\s+сочета\w*|"
                r"экспертиз\w*\s+(?:в|по)|"
                r"компетенц\w*\s+(?:в|по)|"
                r"упомина\w*\s+как|"
                r"продукт\w*|платформ\w*|сервис\w*|услуг\w*|"
                r"направлен\w*|решени\w*|"
                r"offers?|provides?|operates?|owns?|develops?|"
                r"speciali[sz]es?\s+in|works?\s+with|['’]s"
                r")",
                compact,
            )
        )
    return bool(
        re.search(
            r"(?:"
            r"\bот\b|\bby\b|\bfrom\b|принадлеж\w*|разработан\w*|"
            r"управля\w*|продукт\w*|платформ\w*|сервис\w*|"
            r"услуг\w*|направлен\w*|решени\w*"
            r")",
            compact,
        )
    )


def _entity_tail_has_attribution_cue(value: str) -> bool:
    compact = value.casefold().replace("ё", "е")
    return bool(
        re.match(
            r"^[\s—–\-,:]*(?:(?:"
            r"отдельн\w*|самостоятельн\w*|ключев\w*|"
            r"основн\w*|собственн\w*|коммерческ\w*"
            r")\s+)?(?:"
            r"агентств\w*|сервис\w*|услуг\w*|"
            r"платформ\w*|решени\w*|направлен\w*|"
            r"expertise|agency|service|platform"
            r")",
            compact,
        )
    )


_MARKDOWN_HASH_HEADING = re.compile(r"^\s*(#{1,6})\s+", re.UNICODE)
_MARKDOWN_NUMBERED_HEADING = re.compile(
    r"^\s*(?:[-*•]\s*)?(?:\*\*)?\d{1,3}[.)]\s+",
    re.UNICODE,
)
_MARKDOWN_BOLD_HEADING = re.compile(
    r"^\s*\*\*([^*]{2,160})\*\*\s*$",
    re.UNICODE,
)
_MARKDOWN_OWNER_FIELD = re.compile(
    r"(?:застройщик|девелопер|оператор|владелец|"
    r"репутация\s+застройщика|компания|бренд)",
    re.IGNORECASE | re.UNICODE,
)
_MARKDOWN_CARD_FIELD_LABEL = re.compile(
    r"(?:"
    r"почему\s+выигрывает|условия|локация|расположение|"
    r"цена|стоимость|срок(?:и)?|формат|уникальный\s+формат|"
    r"площадь|отделка|качество\s+отделки|архитектура|"
    r"инфраструктура|транспорт|преимущества|недостатки|"
    r"сильная\s+сторона|специализация|экспертиза|компетенции|"
    r"описание|что\s+учесть|важно|итог|вывод|"
    r"график\s+платежей|первоначальный\s+взнос"
    r")",
    re.IGNORECASE | re.UNICODE,
)


def _markdown_colon_label(line: str) -> str:
    """Return a short card label ending in a colon, without Markdown."""

    value = re.sub(r"^\s*[-*•]\s+", "", line).strip()
    value = re.sub(r"[*_`]+", "", value).strip()
    if not value.endswith(":"):
        return ""
    return value[:-1].strip()


def _markdown_colon_label_is_field(label: str) -> bool:
    return bool(_MARKDOWN_CARD_FIELD_LABEL.fullmatch(label.strip()))


def _markdown_heading_signature(line: str) -> tuple[str, int] | None:
    hash_heading = _MARKDOWN_HASH_HEADING.match(line)
    if hash_heading is not None:
        return "hash", len(hash_heading.group(1))
    if _MARKDOWN_NUMBERED_HEADING.match(line) is not None:
        return "numbered", 0
    bold_heading = _MARKDOWN_BOLD_HEADING.match(line)
    if bold_heading is not None:
        # A bold label ending with a colon (for example «Почему выигрывает:»)
        # is a field inside the current card, not a new sibling card.
        if bold_heading.group(1).strip().endswith(":"):
            label = bold_heading.group(1).strip()[:-1].strip()
            return (
                None
                if _markdown_colon_label_is_field(label)
                else ("colon", 0)
            )
        return "bold", 0
    colon_label = _markdown_colon_label(line)
    if colon_label and not _markdown_colon_label_is_field(colon_label):
        return "colon", 0
    stripped = line.strip()
    if (
        stripped
        and re.match(r"^[-*•]", stripped) is None
        and 2 <= len(stripped) <= 120
        and len(stripped.split()) <= 10
        and re.search(r"[.!?;:]$", stripped) is None
        and re.search(r"[A-Za-zА-Яа-яЁё]", stripped) is not None
        and (
            stripped[0].isupper()
            or stripped[0].isdigit()
            or stripped.isupper()
        )
    ):
        return "plain", 0
    return None


def _line_mentions_any_alias(line: str, aliases: list[str]) -> bool:
    return any(_contextual_alias_is_present(line, alias) for alias in aliases)


def _markdown_peer_scope_starts(
    line: str,
    _signature: tuple[str, int] | None,
) -> bool:
    # Any new heading closes the current card.  Matching only heading type or
    # depth allowed hash→numbered and parent→nested competitor leakage.
    return _markdown_heading_signature(line) is not None


def _markdown_owner_field_matches(
    line: str,
    allowed_aliases: list[str],
) -> bool:
    """Accept an owner field only when its value starts with a known owner."""

    match = _MARKDOWN_OWNER_FIELD.search(line)
    if match is None:
        return False
    tail = _normalized_evidence_text(line[match.end() :])
    for alias in allowed_aliases:
        for start, _end in _alias_spans(tail, [alias]):
            if _structured_separator_only(tail[:start]):
                return True
    return False


def _markdown_parent_list_item(
    lines: list[str],
    owner_index: int,
    claim_index: int,
) -> str:
    claim_match = re.match(r"^(\s*)[-*•]\s+", lines[claim_index])
    if claim_match is None:
        return ""
    claim_indent = len(claim_match.group(1).expandtabs(4))
    if claim_indent <= 0:
        return ""
    for index in range(claim_index - 1, owner_index, -1):
        parent_match = re.match(r"^(\s*)[-*•]\s+", lines[index])
        if parent_match is None:
            continue
        parent_indent = len(parent_match.group(1).expandtabs(4))
        if parent_indent < claim_indent:
            return lines[index]
    return ""


def _markdown_list_item_indent(line: str) -> int | None:
    match = re.match(r"^(\s*)[-*•]\s+", line)
    if match is None:
        return None
    return len(match.group(1).expandtabs(4))


def _relation_has_competing_owner(
    relation: str,
    allowed_aliases: list[str],
) -> bool:
    """Reject target comparisons that assign the entity to another actor."""

    normalized = relation.casefold().replace("ё", "е")
    generic_owners = {
        "компания",
        "компании",
        "бренд",
        "бренда",
        "застройщик",
        "застройщика",
        "девелопер",
        "агентство",
    }
    for match in re.finditer(
        r"(?:\bу\b|\bот\b|\bby\b|\bfrom\b)\s+"
        r"([\w.-]{3,}(?:\s+[\w.-]{3,}){0,2}?)"
        r"[^.!?;]{0,80}(?:\bесть\b|предлага\w*|оказыва\w*|"
        r"доступ\w*|представлен\w*|име(?:ет|ют|ла|ли))",
        normalized,
        re.UNICODE,
    ):
        owner = match.group(1)
        if owner in generic_owners:
            continue
        if not any(_alias_is_present(owner, alias) for alias in allowed_aliases):
            return True
    actor_patterns = (
        r"(?:компания|бренд|застройщик|девелопер|агентство)\s+"
        r"([\w.-]{3,}(?:\s+[\w.-]{3,}){0,2})\s+"
        r"(?:предлага\w*|оказыва\w*|предоставля\w*)",
        r"(?:сравнива\w*|сравнен\w*)\s+(?:с|with)\s+"
        r"([\w.-]{3,}(?:\s+[\w.-]{3,}){0,2})",
        r"([\w.-]{3,}(?:\s+[\w.-]{3,}){0,2})\s*,?\s*"
        r"котор\w*\s+(?:предлага\w*|оказыва\w*|предоставля\w*)",
    )
    for pattern in actor_patterns:
        for match in re.finditer(pattern, normalized, re.UNICODE):
            owner = match.group(1).strip()
            if owner in generic_owners:
                continue
            if not any(
                _alias_is_present(owner, alias) for alias in allowed_aliases
            ):
                return True
    return False


def _has_markdown_scoped_target_attribution(
    evidence: str,
    entity_aliases: list[str],
    target_aliases: list[str],
    *,
    direct_target_aliases: list[str] | None = None,
    confirmed_owner_aliases: list[str] | None = None,
) -> bool:
    """Recognize exact same-item and heading-owned Markdown evidence.

    The returned decision never joins text.  It only validates one contiguous
    raw candidate already extracted by ``_literal_target_attribution_evidence``.
    A heading/owner field can scope child bullets until the next peer heading;
    arbitrary paragraph proximity remains insufficient.
    """

    direct_aliases = direct_target_aliases or target_aliases
    known_owners = confirmed_owner_aliases or target_aliases
    lines = evidence.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    for owner_index, owner_line in enumerate(lines):
        if not _line_mentions_any_alias(owner_line, target_aliases):
            continue
        owner_is_direct = _line_mentions_any_alias(owner_line, direct_aliases)

        # A single list item may express the relationship in its second
        # sentence: ``- JOIS (...). В продаже есть квартиры с террасами``.
        if _line_mentions_any_alias(owner_line, entity_aliases):
            normalized_line = _normalized_evidence_text(owner_line)
            for target_start, target_end in _alias_spans(
                normalized_line,
                target_aliases,
            ):
                for entity_start, entity_end in _contextual_alias_spans(
                    normalized_line,
                    entity_aliases,
                ):
                    if target_end <= entity_start:
                        relation = normalized_line[target_end:entity_start]
                        if (
                            len(relation) <= 320
                            and not _relation_has_competing_owner(
                                relation,
                                target_aliases,
                            )
                            and not _structured_field_has_competing_owner(
                                owner_line,
                                entity_aliases,
                                target_aliases,
                            )
                            and (
                                _gap_has_attribution_cue(
                                    relation,
                                    target_first=True,
                                )
                                or (
                                    ":" in relation
                                    and _structured_separator_only(relation)
                                    and re.search(
                                        r"(?:в\s+продаже|есть|доступ\w*|"
                                        r"представлен\w*|предлага\w*|"
                                        r"включа\w*|сда(?:е|ю)тся)",
                                        normalized_line[
                                            entity_end : entity_end + 96
                                        ],
                                        re.UNICODE,
                                    )
                                    is not None
                                )
                                or (
                                    re.match(r"^\s*[-*•]", owner_line)
                                    is not None
                                    and re.search(
                                        r"(?:в\s+продаже|есть|доступ\w*|"
                                        r"представлен\w*|предлага\w*|"
                                        r"включа\w*|формат|услови\w*|"
                                        r"сда(?:е|ю)тся)",
                                        relation,
                                        re.UNICODE,
                                    )
                                    is not None
                                )
                            )
                        ):
                            return True
                    elif entity_end <= target_start:
                        relation = normalized_line[entity_end:target_start]
                        if (
                            len(relation) <= 240
                            and not _relation_has_competing_owner(
                                relation,
                                target_aliases,
                            )
                            and _gap_has_attribution_cue(
                                relation,
                                target_first=False,
                            )
                        ):
                            return True

        signature = _markdown_heading_signature(owner_line)
        parent_indent = _markdown_list_item_indent(owner_line)
        if (
            signature is None
            and parent_indent is not None
            and owner_is_direct
            and owner_line.rstrip().endswith(":")
        ):
            # A target project can be a parent list item rather than a formal
            # heading: ``* JOIS (...):`` followed by indented terms.  Inherit
            # only through strictly deeper child bullets and stop at the next
            # sibling.  This covers MR Group → JOIS → Рассрочка without
            # donating sibling-project properties through the group heading.
            for claim_index in range(owner_index + 1, len(lines)):
                claim_line = lines[claim_index]
                if _markdown_heading_signature(claim_line) is not None:
                    break
                claim_indent = _markdown_list_item_indent(claim_line)
                if claim_indent is None:
                    if claim_line.strip():
                        break
                    continue
                if claim_indent <= parent_indent:
                    break
                if not _line_mentions_any_alias(
                    claim_line,
                    entity_aliases,
                ):
                    continue
                candidate = "\n".join(
                    lines[owner_index : claim_index + 1]
                ).strip()
                if (
                    candidate
                    and len(candidate) <= 1200
                    and not _structured_field_has_competing_owner(
                        claim_line,
                        entity_aliases,
                        target_aliases,
                    )
                    and not _relation_has_competing_owner(
                        claim_line,
                        target_aliases,
                    )
                    and not _relation_has_competing_owner(
                        owner_line,
                        direct_aliases,
                    )
                ):
                    return True
            continue
        # Only an actual heading opens child scope.  A prose line containing
        # «компания» or «бренд» must never become an implicit card heading.
        if signature is None:
            continue

        # Child claims can inherit only a nearby explicit Markdown scope.
        for claim_index in range(owner_index + 1, len(lines)):
            claim_line = lines[claim_index]
            if _markdown_peer_scope_starts(claim_line, signature):
                break
            if (
                _MARKDOWN_OWNER_FIELD.search(claim_line) is not None
                and not _markdown_owner_field_matches(
                    claim_line,
                    known_owners,
                )
            ):
                break
            if not _line_mentions_any_alias(claim_line, entity_aliases):
                continue
            previous_label = (
                _markdown_colon_label(lines[claim_index - 1])
                if claim_index > owner_index + 0
                else ""
            )
            bare_field_continuation = bool(
                previous_label
                and _markdown_colon_label_is_field(previous_label)
                and claim_line.strip()
                and _markdown_heading_signature(claim_line) is None
            )
            if (
                re.match(r"^\s*[-*•]", claim_line) is None
                and not bare_field_continuation
            ):
                continue
            candidate = "\n".join(lines[owner_index : claim_index + 1]).strip()
            parent_item = _markdown_parent_list_item(
                lines,
                owner_index,
                claim_index,
            )
            parent_label = _markdown_colon_label(parent_item)
            parent_is_field = bool(
                parent_label
                and _markdown_colon_label_is_field(parent_label)
            )
            if (
                parent_item
                and not parent_is_field
                and not _line_mentions_any_alias(
                    parent_item,
                    direct_aliases,
                )
            ):
                continue
            if (
                not owner_is_direct
                and not _line_mentions_any_alias(claim_line, direct_aliases)
                and not (
                    parent_item
                    and _line_mentions_any_alias(parent_item, direct_aliases)
                )
            ):
                # A group owner may scope a child claim only along a path that
                # explicitly names this site/project (MR Group → JOIS → terms).
                continue
            if (
                not candidate
                or len(candidate) > 1200
                or _structured_field_has_competing_owner(
                    claim_line,
                    entity_aliases,
                    target_aliases,
                )
                or _relation_has_competing_owner(
                    claim_line,
                    target_aliases,
                )
            ):
                continue
            return True
    return False


def _has_explicit_target_attribution(
    evidence: str,
    entity_aliases: list[str],
    target_aliases: list[str],
    *,
    direct_target_aliases: list[str] | None = None,
    confirmed_owner_aliases: list[str] | None = None,
) -> bool:
    """Reject co-occurrence unless the literal span expresses a relationship."""

    direct_aliases = direct_target_aliases or target_aliases
    if _evidence_excludes_direct_target(evidence, direct_aliases):
        return False
    if _evidence_negates_entity(evidence, entity_aliases):
        return False
    if _scope_qualification_targets_other(evidence, direct_aliases):
        return False
    if _evidence_restricts_claim_to_other_scope(
        evidence,
        direct_aliases,
        entity_aliases,
    ):
        return False
    if _evidence_binds_entity_to_direct_target(
        evidence,
        entity_aliases,
        direct_aliases,
    ):
        return True

    if _has_markdown_scoped_target_attribution(
        evidence,
        entity_aliases,
        target_aliases,
        direct_target_aliases=direct_target_aliases,
        confirmed_owner_aliases=confirmed_owner_aliases,
    ):
        return True

    if _has_structured_target_attribution(
        evidence,
        entity_aliases,
        target_aliases,
    ):
        return True

    lines = evidence.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    # Once the strict Markdown/field parsers decline a multi-line block, the
    # loose prose fallback must not jump over a new card or a competing owner.
    if any(_markdown_heading_signature(line) is not None for line in lines[1:]):
        return False
    owner_aliases = confirmed_owner_aliases or target_aliases
    if any(
        _MARKDOWN_OWNER_FIELD.search(line) is not None
        and not _markdown_owner_field_matches(line, owner_aliases)
        for line in lines[1:]
    ):
        return False

    normalized = _normalized_evidence_text(evidence)
    for entity_alias in entity_aliases:
        if not _contextual_alias_is_present(normalized, entity_alias):
            continue
        for target_alias in target_aliases:
            if not _alias_is_present(normalized, target_alias):
                continue
            if _alias_is_present(entity_alias, target_alias):
                return True
            entity_spans = _contextual_alias_spans(
                normalized,
                [entity_alias],
            )
            target_spans = _alias_spans(normalized, [target_alias])
            for entity_start, entity_end in entity_spans:
                for target_start, target_end in target_spans:
                    if target_end <= entity_start:
                        gap = normalized[target_end:entity_start]
                        if (
                            len(gap) <= 160
                            and not _relation_has_competing_owner(
                                gap,
                                target_aliases,
                            )
                            and not re.search(r"[.!?;\n]", gap)
                            # A labelled field has already been checked by
                            # _has_structured_target_attribution above. Do not
                            # let the looser prose fallback re-attribute a
                            # competitor-owned service from that same field.
                            and not _structured_label_starts_field(gap)
                            and (
                                _gap_has_attribution_cue(
                                    gap,
                                    target_first=True,
                                )
                                or (
                                    _structured_separator_only(gap)
                                    and _entity_tail_has_attribution_cue(
                                        normalized[entity_end : entity_end + 48]
                                    )
                                )
                            )
                        ):
                            return True
                    elif entity_end <= target_start:
                        gap = normalized[entity_end:target_start]
                        if (
                            len(gap) <= 160
                            and not _relation_has_competing_owner(
                                gap,
                                target_aliases,
                            )
                            and not re.search(r"[.!?;\n]", gap)
                            and _gap_has_attribution_cue(
                                gap,
                                target_first=False,
                            )
                        ):
                            return True
    return False


def _evidence_excludes_direct_target(
    evidence: str,
    direct_target_aliases: list[str],
) -> bool:
    """Detect an explicit negation of the analyzed site/project."""

    normalized = _normalized_evidence_text(evidence)
    for alias in direct_target_aliases:
        for start, end in _alias_spans(normalized, [alias]):
            prefix = normalized[max(0, start - 64) : start]
            suffix = normalized[end : end + 96]
            if re.search(
                r"(?:\b(?:но|однако|тогда\s+как)\s+)?"
                r"\bне\b(?:\s+(?:для|у|в|на|про|о))?\s*$|"
                r"\bне\s+включая\s*$|"
                r"\bза\s+исключением\s*$|"
                r"\bкроме\b(?:\s+(?:для|у|в|на))?"
                r"(?:\s+(?:проект\w*|жк|бренд\w*|сайт\w*))?\s*$|"
                r"\b(?:недоступ\w*|отсутств\w*)\s+"
                r"(?:для|у|в|на)\s*$|"
                r"\bexcept(?:\s+(?:for|at|in))?\s*$",
                prefix,
                re.UNICODE,
            ):
                return True
            if re.match(
                r"^\s*(?:(?:пока|еще|больше|никогда)\s+)?"
                r"(?:не|does\s+not|is\s+not|isn't)\s+"
                r"(?:предлага\w*|предоставля\w*|име\w*|"
                r"доступ\w*|поддержива\w*|распространя\w*|"
                r"offer\w*|provide\w*|have\w*)",
                suffix,
                re.UNICODE,
            ):
                return True
            if re.match(
                r"^[^.!?;\n]{0,36}\b(?:не|недоступ\w*|отсутств\w*)\b"
                r"[^.!?;\n]{0,48}(?:предлага\w*|предоставля\w*|"
                r"доступ\w*|поддержива\w*|сервис\w*|услуг\w*|"
                r"решени\w*|продукт\w*)",
                suffix,
                re.UNICODE,
            ):
                return True
            if re.match(
                r"^[^.!?;\n]{0,36}(?:сервис\w*|услуг\w*|"
                r"решени\w*|продукт\w*)[^.!?;\n]{0,36}"
                r"(?:не\s+доступ\w*|недоступ\w*|отсутств\w*)",
                suffix,
                re.UNICODE,
            ):
                return True
            if re.match(
                r"^[^.!?;\n]{0,48}(?:он[аио]?|е[её]|его|их|"
                r"сервис\w*|услуг\w*|решени\w*|продукт\w*)?"
                r"[^.!?;\n]{0,24}(?:нет\b|не\s+доступ\w*|"
                r"недоступ\w*|отсутств\w*|не\s+предусмотр\w*|"
                r"не\s+реализован\w*|не\s+внедрен\w*|"
                r"не\s+применя\w*|не\s+распространя\w*)",
                suffix,
                re.UNICODE,
            ):
                return True
    return False


def _evidence_negates_entity(
    evidence: str,
    entity_aliases: list[str],
) -> bool:
    """Reject explicit absence/negation of the entity itself."""

    normalized = _normalized_evidence_text(evidence)
    for start, end in _contextual_alias_spans(normalized, entity_aliases):
        prefix = normalized[max(0, start - 72) : start]
        suffix = normalized[end : end + 96]
        if re.search(
            r"(?:^|[.:;—–\-])\s*(?:не|без)\s*$",
            prefix,
            re.UNICODE,
        ):
            return True
        if re.match(
            r"^\s*(?:не\s+доступ\w*|недоступ\w*|отсутств\w*|"
            r"не\s+предлага\w*|[—–\-:]?\s*нет\b)",
            suffix,
            re.UNICODE,
        ):
            return True
    return False


def _scope_qualification_targets_other(
    evidence: str,
    direct_target_aliases: list[str],
) -> bool:
    """Detect a local ``only in/at other scope`` qualification."""

    normalized = _normalized_evidence_text(evidence)
    for match in re.finditer(
        r"\b(?:только|лишь|исключительно)\s+"
        r"(?:у|в|для|на)\s+"
        r"(?:(?:проект|жк|жил\w*\s+(?:комплекс|квартал)|"
        r"комплекс|бренд|сайт|компани)\w*\s+)?"
        r"([^.!?;,\n]{2,120})",
        normalized,
        re.UNICODE,
    ):
        scoped_value = match.group(1).strip()
        if not any(
            _alias_is_present(scoped_value, alias)
            for alias in direct_target_aliases
        ):
            return True
    return False


def _evidence_binds_entity_to_direct_target(
    evidence: str,
    entity_aliases: list[str],
    direct_target_aliases: list[str],
) -> bool:
    """Accept an explicit entity → analyzed project/site binding."""

    normalized = _normalized_evidence_text(evidence)
    entity_spans = _contextual_alias_spans(normalized, entity_aliases)
    target_spans = _alias_spans(normalized, direct_target_aliases)
    for entity_start, entity_end in entity_spans:
        for target_start, target_end in target_spans:
            if entity_end <= target_start:
                relation = normalized[entity_end:target_start]
                if re.search(
                    r"\b(?:специально\s+|именно\s+)?"
                    r"(?:в|для|на)\s+"
                    r"(?:(?:проект\w*|жк|"
                    r"жил\w*\s+(?:комплекс|квартал)\w*|"
                    r"комплекс\w*|бренд\w*|сайт\w*)\s+)?$",
                    relation,
                    re.UNICODE,
                ):
                    return True
            elif target_end <= entity_start:
                relation = normalized[target_end:entity_start]
                if re.fullmatch(
                    r"\s*(?:(?:,?\s*(?:и|and)\s+)"
                    r"[^.!?;\n]{2,96}?\s+)?"
                    r"(?:предлага\w*|предоставля\w*|поддержива\w*|"
                    r"использу\w*|реализу\w*|име\w*)\s*",
                    relation,
                    re.UNICODE,
                ):
                    # ``JOIS and City Bay support White Box`` is still an
                    # explicit positive claim about JOIS.  The coordinated
                    # sibling must not turn the whole subject into a
                    # competing-owner false negative.
                    return True
    return False


def _evidence_restricts_claim_to_other_scope(
    evidence: str,
    direct_target_aliases: list[str],
    entity_aliases: list[str],
) -> bool:
    """Reject owner-level claims explicitly limited to another project."""

    normalized = _normalized_evidence_text(evidence)
    for entity_start, entity_end in _contextual_alias_spans(
        normalized,
        entity_aliases,
    ):
        vicinity_start = max(0, entity_start - 120)
        vicinity_end = min(len(normalized), entity_end + 260)
        vicinity = normalized[vicinity_start:vicinity_end]
        if re.search(
            r"\b(?:не\s+для|не\s+в|не\s+у)\s+"
            r"(?:сво(?:его|ем)|эт(?:ого|ом)|данн(?:ого|ом))\s+"
            r"(?:проект\w*|жк|бренд\w*|сайт\w*)",
            vicinity,
            re.UNICODE,
        ):
            return True

        for binding in re.finditer(
            r"\b(?:только|лишь|исключительно)?\s*"
            r"(?:в|для|на)\s+"
            r"(?:проект\w*|жк|жил\w*\s+комплекс\w*|"
            r"жил\w*\s+квартал\w*|комплекс\w*|"
            r"бренд\w*|сайт\w*)\s+"
            r"([^.!?;,\n]{2,100})",
            vicinity,
            re.UNICODE,
        ):
            bound_scope = binding.group(1).strip()
            if not any(
                _alias_is_present(bound_scope, alias)
                for alias in direct_target_aliases
            ):
                return True

        owner_restriction = re.search(
            r"\b(?:только|лишь|исключительно)\s+у\s+"
            r"([^.!?;,\n]{2,100})",
            vicinity,
            re.UNICODE,
        )
        if owner_restriction is not None and not any(
            _alias_is_present(owner_restriction.group(1), alias)
            for alias in direct_target_aliases
        ):
            return True

    for entity_start, entity_end in _contextual_alias_spans(
        evidence,
        entity_aliases,
    ):
        # Without an explicit scope noun, accept only a multi-token proper
        # name (``City Bay``).  This avoids treating ``только в приложении``
        # or ``только в Москве`` as a sibling-product binding.  Raw offsets
        # are computed separately because normalized text collapses Markdown
        # and whitespace and therefore cannot index the original string.
        original_vicinity = evidence[
            max(0, entity_start - 120) : min(len(evidence), entity_end + 260)
        ]
        original_prefix = evidence[max(0, entity_start - 220) : entity_start]
        exclusive_subject = re.search(
            r"(?:^|[.!?;\n])\s*"
            r"(?:только|лишь|исключительно)\s+"
            r"([^.!?;\n]{2,140}?)\s+"
            r"(?:предлага\w*|предоставля\w*|поддержива\w*|"
            r"име\w*|использу\w*|реализу\w*)\s*$",
            original_prefix,
            re.IGNORECASE | re.UNICODE,
        )
        if exclusive_subject is not None and not any(
            _alias_is_present(exclusive_subject.group(1), alias)
            for alias in direct_target_aliases
        ):
            return True
        named_restriction = re.search(
            r"\b(?:(?:только|лишь|исключительно)\s+)?"
            r"(?:в|для|на)\s+"
            r"([A-Z][A-Za-z0-9-]+(?:\s+[A-Z][A-Za-z0-9-]+){1,3})",
            original_vicinity,
            re.UNICODE,
        )
        if named_restriction is not None and not any(
            _alias_is_present(named_restriction.group(1), alias)
            for alias in direct_target_aliases
        ):
            return True

        # ``Service: only City Bay and Symphony 34`` limits an owner-level
        # statement to named sibling products even without ``in/for``.  Keep
        # the match conservative: the first scope must be a multi-token proper
        # name, so ordinary constraints such as ``only on weekdays`` do not
        # become product ownership rules.
        suffix = evidence[entity_end : min(len(evidence), entity_end + 220)]
        bare_named_restriction = re.match(
            r"^[\s:—–\-,()]{0,24}"
            r"(?:доступ\w*\s+|предлага\w*\s+|предусмотр\w*\s+)?"
            r"(?:только|лишь|исключительно)\s+"
            r"(?:у\s+|в\s+|для\s+|на\s+)?"
            r"([A-ZА-ЯЁ][A-Za-zА-Яа-яЁё0-9-]+"
            r"(?:\s+[A-ZА-ЯЁ0-9][A-Za-zА-Яа-яЁё0-9-]*){1,3})",
            suffix,
            re.UNICODE,
        )
        if bare_named_restriction is not None and not any(
            _alias_is_present(
                bare_named_restriction.group(1),
                alias,
            )
            for alias in direct_target_aliases
        ):
            return True
    return False


def _entity_local_claim_context(
    answer_text: str,
    entity_start: int,
    entity_end: int,
    direct_target_aliases: list[str],
    entity_aliases: list[str],
) -> str:
    """Return a raw local claim plus its immediate qualification."""

    line_start = answer_text.rfind("\n", 0, entity_start) + 1
    line_end = answer_text.find("\n", entity_end)
    if line_end < 0:
        line_end = len(answer_text)
    clause_start = max(
        [line_start]
        + [
            position + 1
            for mark in ".!?;"
            if (position := answer_text.rfind(mark, line_start, entity_start))
            >= 0
        ]
    )
    clause_ends = [
        position + 1
        for mark in ".!?;"
        if (position := answer_text.find(mark, entity_end, line_end)) >= 0
    ]
    clause_end = min(clause_ends) if clause_ends else line_end
    claim = answer_text[clause_start:clause_end].strip()

    def is_negative_qualification(value: str) -> bool:
        normalized = _normalized_evidence_text(value)
        direct_negative = bool(
            any(
                _alias_is_present(normalized, alias)
                for alias in direct_target_aliases
            )
            and re.search(
                r"\b(?:не|нет|недоступ\w*|отсутств\w*|"
                r"исключ\w*|кроме|не\s+распространя\w*)\b",
                normalized,
                re.UNICODE,
            )
        )
        competing_owner = _structured_field_has_competing_owner(
            value,
            entity_aliases,
            direct_target_aliases,
        )
        contrast_or_exclusive = re.search(
            r"\b(?:но|однако|только|лишь|исключительно|"
            r"эксклюзив\w*|принадлеж\w*|разработ\w*|создан\w*)\b",
            normalized,
            re.UNICODE,
        ) is not None
        return bool(
            direct_negative
            or _scope_qualification_targets_other(
                value,
                direct_target_aliases,
            )
            or _evidence_negates_entity(value, entity_aliases)
            or (competing_owner and contrast_or_exclusive)
        )

    # Keep one immediately following negative qualification, but never let a
    # new competitor sentence or sibling bullet poison an already valid claim.
    remainder_end = line_end
    next_clause_ends = [
        position + 1
        for mark in ".!?;"
        if (position := answer_text.find(mark, clause_end, remainder_end)) >= 0
    ]
    next_clause_end = min(next_clause_ends) if next_clause_ends else remainder_end
    next_clause = answer_text[clause_end:next_clause_end].strip()
    if next_clause and is_negative_qualification(next_clause):
        claim = f"{claim} {next_clause}".strip()

    if line_end < len(answer_text):
        cursor = line_end + 1
        # Skip blank separators and a short neutral continuation, but stop at
        # a new card/bullet or a repeated entity claim.  This catches
        # ``Условия уточняются`` followed by a direct qualification without
        # borrowing an unrelated competitor block.
        for _line_offset in range(5):
            next_line_end = answer_text.find("\n", cursor)
            if next_line_end < 0:
                next_line_end = len(answer_text)
            next_line = answer_text[cursor:next_line_end].strip()
            cursor = next_line_end + 1
            if not next_line:
                if next_line_end >= len(answer_text):
                    break
                continue
            if is_negative_qualification(next_line):
                claim = f"{claim}\n{next_line}".strip()
                break
            if (
                re.match(r"^\s*(?:#{1,6}|[-*•])", next_line)
                or _line_mentions_any_alias(next_line, entity_aliases)
            ):
                break
            if next_line_end >= len(answer_text):
                break
    return claim[:1200]


def _attribution_pair_crosses_scope_boundary(
    answer_text: str,
    *,
    entity_start: int,
    target_start: int,
    target_aliases: list[str],
) -> bool:
    """Reject a target/entity pair that crosses into another result card.

    Valid multi-line inheritance is directional: an explicit target owner may
    scope a later child claim.  A service mentioned before the target cannot
    become target-owned retroactively, and a new Markdown/list owner between
    the two offsets closes the scope.  Keeping this check pair-aware prevents
    another entity occurrence elsewhere in a long candidate from bypassing a
    STOP rule during global evidence validation.
    """

    target_line_start = answer_text.rfind("\n", 0, target_start) + 1
    target_line_end = answer_text.find("\n", target_start)
    if target_line_end < 0:
        target_line_end = len(answer_text)
    entity_line_start = answer_text.rfind("\n", 0, entity_start) + 1
    if target_line_start == entity_line_start:
        return False
    if entity_start < target_start:
        return True

    target_line = answer_text[target_line_start:target_line_end]
    target_indent = _markdown_list_item_indent(target_line)
    entity_line_end = answer_text.find("\n", entity_start)
    if entity_line_end < 0:
        entity_line_end = len(answer_text)
    entity_line = answer_text[entity_line_start:entity_line_end]
    entity_indent = _markdown_list_item_indent(entity_line)

    cursor = target_line_end + 1
    while cursor < entity_line_start:
        line_end = answer_text.find("\n", cursor)
        if line_end < 0 or line_end > entity_line_start:
            line_end = entity_line_start
        line = answer_text[cursor:line_end]
        cursor = line_end + 1
        if not line.strip():
            continue
        line_mentions_target = _line_mentions_any_alias(
            line,
            target_aliases,
        )
        if (
            _markdown_heading_signature(line) is not None
            and not line_mentions_target
        ):
            return True
        if (
            _MARKDOWN_OWNER_FIELD.search(line) is not None
            and not _markdown_owner_field_matches(line, target_aliases)
        ):
            return True

        line_indent = _markdown_list_item_indent(line)
        if line_indent is None or line_mentions_target:
            continue
        label = _markdown_colon_label(line)
        is_field = bool(
            label and _markdown_colon_label_is_field(label)
        )
        if is_field:
            continue
        if target_indent is not None and line_indent <= target_indent:
            return True
        if entity_indent is not None and line_indent < entity_indent:
            return True
    return False


def _attribution_pair_is_disqualified(
    answer_text: str,
    *,
    entity_start: int,
    entity_end: int,
    target_start: int,
    entity_aliases: list[str],
    target_aliases: list[str],
    direct_target_aliases: list[str],
    confirmed_owner_aliases: list[str] | None = None,
) -> bool:
    """Apply scope STOP rules before any evidence shortening."""

    claim = _entity_local_claim_context(
        answer_text,
        entity_start,
        entity_end,
        direct_target_aliases,
        entity_aliases,
    )
    direct_binding = _evidence_binds_entity_to_direct_target(
        claim,
        entity_aliases,
        direct_target_aliases,
    )
    competing_owner = _structured_field_has_competing_owner(
        claim,
        entity_aliases,
        target_aliases,
    )
    later_competing_qualification = False
    claim_entity_spans = _contextual_alias_spans(claim, entity_aliases)
    if claim_entity_spans:
        first_entity_end = claim_entity_spans[0][1]
        boundaries = [
            position + 1
            for position, character in enumerate(claim[first_entity_end:])
            if character in ".!?;\n"
        ]
        if boundaries:
            tail_start = first_entity_end + min(boundaries)
            qualification_tail = claim[tail_start:].strip()
            later_competing_qualification = bool(
                qualification_tail
                and _structured_field_has_competing_owner(
                    qualification_tail,
                    entity_aliases,
                    target_aliases,
                )
            )
    return bool(
        _attribution_pair_crosses_scope_boundary(
            answer_text,
            entity_start=entity_start,
            target_start=target_start,
            target_aliases=(confirmed_owner_aliases or target_aliases),
        )
        or _evidence_excludes_direct_target(claim, direct_target_aliases)
        or _evidence_negates_entity(claim, entity_aliases)
        or _scope_qualification_targets_other(
            claim,
            direct_target_aliases,
        )
        or _evidence_restricts_claim_to_other_scope(
            claim,
            direct_target_aliases,
            entity_aliases,
        )
        or (
            competing_owner
            and (not direct_binding or later_competing_qualification)
        )
    )


def _literal_target_attribution_evidence(
    answer_text: str,
    entity_aliases: list[str],
    target_aliases: list[str],
    *,
    direct_target_aliases: list[str] | None = None,
    confirmed_owner_aliases: list[str] | None = None,
) -> str:
    """Extract the shortest exact local block that proves attribution."""

    entity_spans = _contextual_alias_spans(answer_text, entity_aliases)
    target_spans = _alias_spans(answer_text, target_aliases)
    if not entity_spans or not target_spans:
        return ""

    candidates: set[str] = set()
    text_length = len(answer_text)
    hard_boundaries = ".!?;\n"
    direct_aliases = direct_target_aliases or target_aliases
    for entity_start, entity_end in entity_spans:
        for target_start, target_end in target_spans:
            pair_start = min(entity_start, target_start)
            pair_end = max(entity_end, target_end)
            if pair_end - pair_start > 1200:
                continue
            if _attribution_pair_is_disqualified(
                answer_text,
                entity_start=entity_start,
                entity_end=entity_end,
                target_start=target_start,
                entity_aliases=entity_aliases,
                target_aliases=target_aliases,
                direct_target_aliases=direct_aliases,
                confirmed_owner_aliases=confirmed_owner_aliases,
            ):
                continue

            # Candidate evidence must preserve the complete surrounding
            # sentence/line.  Pair-only slices used to hide disqualifying
            # suffixes such as ``только в City Bay`` or ``но не для JOIS``.
            starts: set[int] = set()
            ends: set[int] = set()
            for boundary in hard_boundaries:
                previous = answer_text.rfind(boundary, 0, pair_start)
                if previous >= 0:
                    starts.add(previous + 1)
                following = answer_text.find(boundary, pair_end)
                if following >= 0:
                    ends.add(following + 1)
                    second = answer_text.find(boundary, following + 1)
                    if second >= 0:
                        ends.add(second + 1)

            paragraph_start = answer_text.rfind("\n\n", 0, pair_start)
            starts.add(0 if paragraph_start < 0 else paragraph_start + 2)
            paragraph_end = answer_text.find("\n\n", pair_end)
            ends.add(text_length if paragraph_end < 0 else paragraph_end)

            line_start = answer_text.rfind("\n", 0, pair_start)
            starts.add(0 if line_start < 0 else line_start + 1)
            line_end = answer_text.find("\n", pair_end)
            if line_end < 0:
                ends.add(text_length)
            else:
                ends.add(line_end)
                next_line_end = answer_text.find("\n", line_end + 1)
                ends.add(
                    text_length
                    if next_line_end < 0
                    else next_line_end
                )

            for start in starts:
                for end in ends:
                    if start > pair_start or end < pair_end:
                        continue
                    candidate = answer_text[start:end].strip()
                    if (
                        not candidate
                        or len(candidate) > 1200
                    ):
                        continue
                    candidates.add(candidate)

    for candidate in sorted(
        candidates,
        key=lambda value: (len(value.split()), len(value)),
    ):
        if _has_explicit_target_attribution(
            candidate,
            entity_aliases,
            target_aliases,
            direct_target_aliases=direct_target_aliases,
            confirmed_owner_aliases=confirmed_owner_aliases,
        ):
            return candidate
    return ""


def _portfolio_entity_is_grounded(
    answer_text: str,
    entity: dict[str, Any],
    target_aliases: list[str],
    mention: dict[str, Any] | None = None,
    *,
    direct_target_aliases: list[str] | None = None,
    confirmed_owner_aliases: list[str] | None = None,
) -> bool:
    """Require a literal name and explicit evidence for generic offerings."""

    if not answer_text:
        return False
    configured_aliases = entity.get("_attribution_aliases")
    if isinstance(configured_aliases, list) and configured_aliases:
        target_aliases = [str(value) for value in configured_aliases]
    configured_direct = entity.get("_direct_target_aliases")
    if direct_target_aliases is None and isinstance(configured_direct, list):
        direct_target_aliases = [str(value) for value in configured_direct]
    configured_owners = entity.get("_confirmed_owner_aliases")
    if confirmed_owner_aliases is None and isinstance(configured_owners, list):
        confirmed_owner_aliases = [str(value) for value in configured_owners]
    matching_aliases = [
        (alias, policy)
        for alias, policy in _entity_alias_entries(
            entity,
            excluded_aliases=target_aliases,
        )
        if (
            _alias_is_present(answer_text, alias)
            if policy == "standalone"
            else _contextual_alias_is_present(answer_text, alias)
        )
    ]
    if not matching_aliases:
        return False
    if any(policy == "standalone" for _alias, policy in matching_aliases):
        return True

    contextual_aliases = [
        alias
        for alias, policy in matching_aliases
        if policy == "requires_target_attribution"
    ]
    if any(
        any(
            _alias_is_present(entity_alias, target_alias)
            for target_alias in target_aliases
        )
        for entity_alias in contextual_aliases
    ):
        return True

    evidence = str((mention or {}).get("evidence") or "").strip()
    return bool(
        (mention or {}).get("attributed_to_target") is True
        and _evidence_is_literal(answer_text, evidence)
        and _has_explicit_target_attribution(
            evidence,
            contextual_aliases,
            target_aliases,
            direct_target_aliases=direct_target_aliases,
            confirmed_owner_aliases=confirmed_owner_aliases,
        )
    )


def _explicit_profile_portfolio_facts(
    answer_text: str,
    profile: dict[str, Any],
    catalog: dict[str, Any],
) -> list[str]:
    """Return profile-confirmed offerings explicitly bound to the target."""

    direct_aliases = _target_aliases(profile, catalog)
    confirmed_owner_aliases = _attribution_owner_aliases(profile, catalog)
    attribution_alias_map = _entity_attribution_alias_map(profile, catalog)
    confirmed: list[str] = []
    for entity in catalog.get("entities") or []:
        if (
            not isinstance(entity, dict)
            or not _catalog_marks_portfolio_entity(entity)
            or not _profile_confirms_portfolio_entity(entity, profile)
        ):
            continue
        canonical = str(entity.get("canonical_name") or "").strip()
        attribution_aliases = attribution_alias_map.get(
            canonical.casefold(),
            direct_aliases,
        )
        entity_aliases = [
            alias
            for alias, _policy in _entity_alias_entries(
                entity,
                profile,
                excluded_aliases=attribution_aliases,
            )
            if _contextual_alias_is_present(answer_text, alias)
        ]
        if not entity_aliases:
            continue
        if _literal_target_attribution_evidence(
            answer_text,
            entity_aliases,
            attribution_aliases,
            direct_target_aliases=direct_aliases,
            confirmed_owner_aliases=confirmed_owner_aliases,
        ):
            confirmed.append(canonical)
    return confirmed


def _reconcile_annotation(
    item: dict[str, Any],
    pending_answer: dict[str, Any],
    profile: dict[str, Any],
    catalog: dict[str, Any],
    *,
    annotation_input_sha256: str = "",
) -> dict[str, Any]:
    """Enforce literal entity evidence before deterministic metrics are computed."""

    reconciled = dict(item)
    answer_text = str(pending_answer.get("answer") or "")
    mentions = [
        dict(value)
        for value in reconciled.get("entity_mentions") or []
        if isinstance(value, dict)
    ]
    existing_names = {
        str(value.get("canonical_name") or "").casefold()
        for value in mentions
    }
    existing_mentions = {
        str(value.get("canonical_name") or "").casefold(): value
        for value in mentions
        if value.get("canonical_name")
    }
    reconciliation_notes: list[str] = []
    direct_aliases = _target_aliases(profile, catalog)
    confirmed_owner_aliases = _attribution_owner_aliases(profile, catalog)
    entity_attribution_aliases = _entity_attribution_alias_map(
        profile,
        catalog,
    )
    for entity in catalog.get("entities") or []:
        if not isinstance(entity, dict):
            continue
        canonical = str(entity.get("canonical_name") or "").strip()
        attribution_aliases = entity_attribution_aliases.get(
            canonical.casefold(),
            direct_aliases,
        )
        is_portfolio_candidate = _catalog_marks_portfolio_entity(entity)
        if (
            is_portfolio_candidate
            and not _profile_confirms_portfolio_entity(entity, profile)
        ):
            continue
        matching_aliases = [
            (alias, policy)
            for alias, policy in _entity_alias_entries(
                entity,
                profile,
                excluded_aliases=attribution_aliases,
            )
            if (
                _alias_is_present(answer_text, alias)
                if policy == "standalone"
                else _contextual_alias_is_present(answer_text, alias)
            )
        ]
        if is_portfolio_candidate:
            contextual_aliases = [
                alias
                for alias, policy in matching_aliases
                if policy == "requires_target_attribution"
            ]
            literal_attribution = _literal_target_attribution_evidence(
                answer_text,
                contextual_aliases,
                attribution_aliases,
                direct_target_aliases=direct_aliases,
                confirmed_owner_aliases=confirmed_owner_aliases,
            )
            if canonical and literal_attribution:
                existing = existing_mentions.get(canonical.casefold())
                if existing is None:
                    existing = {
                        "canonical_name": canonical,
                        "position": None,
                        "role": "mentioned",
                        "attributed_to_target": True,
                        "evidence": literal_attribution,
                    }
                    mentions.append(existing)
                    existing_names.add(canonical.casefold())
                    existing_mentions[canonical.casefold()] = existing
                else:
                    existing["attributed_to_target"] = True
                    existing["evidence"] = literal_attribution
                reconciliation_notes.append(
                    "Детерминированно подтверждена буквальная связь "
                    f"с целевым брендом: {canonical}."
                )
                continue

            # Without an explicit literal relationship, reconciliation may
            # repair only a distinctive standalone product name. Generic
            # co-occurrence remains insufficient.
            matching_aliases = [
                (alias, policy)
                for alias, policy in matching_aliases
                if policy == "standalone"
            ]
        matched_alias_evidence = _literal_alias_evidence(
            answer_text,
            matching_aliases,
        )
        if (
            canonical
            and canonical.casefold() not in existing_names
            and matched_alias_evidence
        ):
            mentions.append(
                {
                    "canonical_name": canonical,
                    "position": None,
                    "role": "mentioned",
                    "attributed_to_target": False,
                    "evidence": matched_alias_evidence,
                }
            )
            existing_names.add(canonical.casefold())
            reconciliation_notes.append(
                f"Детерминированно найдено буквальное упоминание: {canonical}."
            )

    entities_by_name = {
        str(entity.get("canonical_name") or "").casefold(): entity
        for entity in catalog.get("entities") or []
        if isinstance(entity, dict) and entity.get("canonical_name")
    }
    for mention in mentions:
        if mention.get("attributed_to_target") is not True:
            continue
        entity = entities_by_name.get(
            str(mention.get("canonical_name") or "").casefold()
        ) or {}
        attribution_aliases = entity_attribution_aliases.get(
            str(entity.get("canonical_name") or "").casefold(),
            direct_aliases,
        )
        relationship = str(
            entity.get("target_relationship")
            or entity.get("relationship")
            or ""
        ).casefold()
        is_exact_target = (
            entity.get("category") == "target"
            and relationship in {"exact_target", "self"}
        )
        if is_exact_target:
            continue
        if (
            not _catalog_marks_portfolio_entity(entity)
            or not _profile_confirms_portfolio_entity(entity, profile)
        ):
            mention["attributed_to_target"] = False
            reconciliation_notes.append(
                "Снята атрибуция сущности вне подтверждённого портфеля: "
                f"{mention.get('canonical_name')}."
            )
            continue
        if (
            _portfolio_entity_is_grounded(
                answer_text,
                entity,
                attribution_aliases,
                mention,
                direct_target_aliases=direct_aliases,
                confirmed_owner_aliases=confirmed_owner_aliases,
            )
        ):
            continue
        contextual_aliases = [
            alias
            for alias, policy in _entity_alias_entries(
                entity,
                profile,
                excluded_aliases=attribution_aliases,
            )
            if policy == "requires_target_attribution"
            and _contextual_alias_is_present(answer_text, alias)
        ]
        literal_evidence = _literal_target_attribution_evidence(
            answer_text,
            contextual_aliases,
            attribution_aliases,
            direct_target_aliases=direct_aliases,
            confirmed_owner_aliases=confirmed_owner_aliases,
        )
        if literal_evidence:
            mention["evidence"] = literal_evidence
            reconciliation_notes.append(
                "Детерминированно восстановлен буквальный фрагмент "
                f"атрибуции: {entity.get('canonical_name')}."
            )
        else:
            mention["attributed_to_target"] = False
            reconciliation_notes.append(
                "Снята атрибуция без проверяемого буквального фрагмента: "
                f"{entity.get('canonical_name')}."
            )

    # One deterministic choke point governs every persisted quote.  The LLM
    # may change case, Markdown or punctuation even after being asked to copy;
    # only an exact raw slice containing a complete declared alias survives.
    sanitized_mentions: list[dict[str, Any]] = []
    for mention in mentions:
        entity = entities_by_name.get(
            str(mention.get("canonical_name") or "").casefold()
        )
        if not entity:
            continue
        is_portfolio_candidate = _catalog_marks_portfolio_entity(entity)
        attribution_aliases = entity_attribution_aliases.get(
            str(entity.get("canonical_name") or "").casefold(),
            direct_aliases,
        )
        alias_entries = _entity_alias_entries(
            entity,
            profile,
            excluded_aliases=(
                attribution_aliases if is_portfolio_candidate else []
            ),
        )
        matching_alias_entries = [
            (alias, policy)
            for alias, policy in alias_entries
            if (
                _alias_is_present(answer_text, alias)
                if policy == "standalone"
                else _contextual_alias_is_present(answer_text, alias)
            )
        ]
        if not matching_alias_entries:
            continue

        contextual_aliases = [
            alias
            for alias, policy in matching_alias_entries
            if policy == "requires_target_attribution"
        ]
        attribution_evidence = (
            _literal_target_attribution_evidence(
                answer_text,
                contextual_aliases,
                attribution_aliases,
                direct_target_aliases=direct_aliases,
                confirmed_owner_aliases=confirmed_owner_aliases,
            )
            if is_portfolio_candidate and contextual_aliases
            else ""
        )
        if attribution_evidence:
            mention["attributed_to_target"] = True
            mention["evidence"] = attribution_evidence
        else:
            if is_portfolio_candidate and contextual_aliases:
                mention["attributed_to_target"] = False
            evidence = str(mention.get("evidence") or "").strip()
            if not (
                _evidence_is_literal(answer_text, evidence)
                and _evidence_contains_complete_alias(
                    evidence,
                    matching_alias_entries,
                )
            ):
                evidence = _literal_alias_evidence(
                    answer_text,
                    matching_alias_entries,
                )
                if evidence:
                    reconciliation_notes.append(
                        "Доказательный фрагмент восстановлен точным срезом "
                        f"raw-ответа: {entity.get('canonical_name')}."
                    )
            mention["evidence"] = evidence

        if not _evidence_is_literal(
            answer_text,
            str(mention.get("evidence") or ""),
        ):
            continue
        sanitized_mentions.append(mention)

    reconciled["entity_mentions"] = sanitized_mentions
    reconciled["evidence"] = list(
        dict.fromkeys(
            evidence.strip()
            for raw_evidence in reconciled.get("evidence") or []
            if isinstance(raw_evidence, str)
            and (evidence := raw_evidence.strip())
            and _evidence_is_literal(answer_text, evidence)
        )
    )

    direct_alias_present = any(
        _alias_is_present(answer_text, alias) for alias in direct_aliases
    )
    if direct_alias_present:
        if reconciled.get("target_mentioned") is not True:
            reconciliation_notes.append(
                "Детерминированно найден прямой алиас целевого бренда."
            )
        reconciled["target_mentioned"] = True
        if reconciled.get("target_role") in {None, "absent", "unknown"}:
            reconciled["target_role"] = "mentioned"
        if reconciled.get("sentiment") == "unknown":
            reconciled["sentiment"] = "neutral"
    else:
        if reconciled.get("target_mentioned") is True:
            reconciliation_notes.append(
                "Снято неподтверждённое упоминание целевого бренда: "
                "его имени или алиаса нет в исходном ответе."
            )
        reconciled["target_mentioned"] = False
        reconciled["target_position"] = None
        reconciled["target_role"] = "absent"
        reconciled["sentiment"] = "unknown"

        exact_target_names = {
            str(entity.get("canonical_name") or "").casefold()
            for entity in catalog.get("entities") or []
            if isinstance(entity, dict)
            and entity.get("category") == "target"
            and str(
                entity.get("target_relationship")
                or entity.get("relationship")
                or ""
            ).casefold()
            in {"exact_target", "self"}
        }
        reconciled["entity_mentions"] = [
            mention
            for mention in reconciled["entity_mentions"]
            if str(mention.get("canonical_name") or "").casefold()
            not in exact_target_names
        ]
        reconciled["evidence"] = [
            evidence
            for evidence in reconciled.get("evidence") or []
            if not any(
                _alias_is_present(str(evidence), alias)
                for alias in direct_aliases
            )
        ]

    brand_answer = reconciled.get("brand_answer")
    if not isinstance(brand_answer, dict):
        brand_answer = {
            "directness": "not_applicable",
            "specificity": "not_applicable",
            "supported_facets": [],
            "contradictions": [],
        }
    scenario_role = str(pending_answer.get("scenario_role") or "")
    if scenario_role == "unbranded_discovery":
        brand_answer = {
            "directness": "not_applicable",
            "specificity": "not_applicable",
            "supported_facets": [],
            "contradictions": [],
        }
    elif scenario_role == "brand_diagnostic" and (
        brand_answer.get("directness") == "not_applicable"
        or brand_answer.get("specificity") == "not_applicable"
    ):
        explicit_portfolio_facts = _explicit_profile_portfolio_facts(
            answer_text,
            profile,
            catalog,
        )
        if direct_alias_present and explicit_portfolio_facts:
            supported_facets = list(
                dict.fromkeys(
                    [
                        *[
                            str(value)
                            for value in brand_answer.get(
                                "supported_facets"
                            )
                            or []
                            if value
                        ],
                        "offering",
                        "portfolio",
                    ]
                )
            )
            brand_answer = {
                **brand_answer,
                "directness": "partial",
                "specificity": "specific",
                "supported_facets": supported_facets,
                "contradictions": list(
                    brand_answer.get("contradictions") or []
                ),
            }
            reconciliation_notes.append(
                "Исправлен недопустимый not_applicable в брендовом "
                "сценарии: raw содержит явную связь цели с "
                "подтверждённым предложением ("
                + ", ".join(explicit_portfolio_facts)
                + ")."
            )
    reconciled["brand_answer"] = brand_answer
    reconciled["_annotation_version"] = ANNOTATION_VERSION
    reconciled["_answer_sha256"] = str(pending_answer.get("answer_sha256") or "")
    reconciled["_answer_model"] = str(pending_answer.get("answer_model") or "")
    reconciled["_annotation_input_sha256"] = annotation_input_sha256
    reconciled["_reconciliation_notes"] = reconciliation_notes
    return reconciled


def _annotation_context_sha256(
    profile: dict[str, Any],
    catalog: dict[str, Any],
    research_guidance: str = "",
) -> str:
    context = {
        "brand_name": profile.get("brand_name"),
        "site_profile": profile,
        "target_aliases": _target_aliases(profile, catalog),
        "entity_catalog": catalog.get("entities") or [],
        "entity_attribution_aliases": _entity_attribution_alias_map(
            profile,
            catalog,
        ),
        "research_guidance": research_guidance.strip(),
    }
    return hashlib.sha256(
        json.dumps(
            context,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


async def _annotate_answers(
    run_id: str,
    profile: dict[str, Any],
    catalog: dict[str, Any],
    *,
    research_guidance: str = "",
    _completion_attempt: int = 1,
) -> None:
    annotation_input_sha256 = _annotation_context_sha256(
        profile,
        catalog,
        research_guidance,
    )
    pending = await _unannotated_answers(
        run_id,
        annotation_input_sha256=annotation_input_sha256,
    )
    if not pending:
        return
    total = len(pending)
    system = f"""
Разметь каждый ответ атомарно и только по его тексту.
Целевая сущность: {profile.get("brand_name")}.
Используй каталог канонических названий. Не вычисляй доли, индексы, оценки
видимости и другие метрики.

Правила:
- valid=false только для отказа, пустого или нерелевантного ответа;
- target_mentioned=true только при явном названии цели или её алиаса;
- position — место в явном списке рекомендаций, иначе null;
- recommended означает прямую рекомендацию, conditional — рекомендацию с
условием, mentioned — нейтральное упоминание, excluded — явное исключение;
- если цель не названа, role=absent и sentiment=unknown;
- entity_mentions включает как конкурентов, так и продукты и бизнес-направления
  цели из каталога; не пропускай буквальные названия target-сущностей.
  Для каждой сущности скопируй evidence как один точный непрерывный фрагмент
  raw-ответа. Сохраняй регистр, орфографию, пунктуацию, Markdown-маркеры и
  переносы строк. Нельзя добавлять кавычки или многоточия, исправлять опечатки,
  обрезать слово/название бренда либо склеивать заголовок и дочерний пункт.
  attributed_to_target=true только когда этот фрагмент явно связывает сущность
  с целевым брендом. Простая близость в списке, общая тема ответа, ссылка на
  сайт или упоминание услуги у другого агентства атрибуцией не считаются.
  Для общей услуги target-портфеля evidence должен быть одним непрерывным
  точным фрагментом до 1200 знаков и содержать одновременно разрешённого для
  этой сущности владельца из entity_attribution_aliases, алиас услуги и слова,
  выражающие их связь. Если связь задана заголовком Markdown и дочерним
  пунктом, не склеивай их: верни точный
  дочерний фрагмент и attributed_to_target=false — структурный контекст затем
  проверит детерминированный слой. Если точного фрагмента нет,
  attributed_to_target=false;
- для brand_diagnostic отдельно оцени brand_answer. directness показывает,
  ответила ли модель на вопрос. specificity=specific только когда в ответе есть
  конкретные, проверяемые сведения о самой цели, подтверждённые site_profile
  или entity_catalog; повтор названия из вопроса ничего не доказывает.
  generic — только общие слова без конкретики, none — знания нет,
  contradictory — ответ противоречит переданным фактам. Перечисли только
  подтверждённые facets и коротко запиши противоречия. Для brand_diagnostic
  значения directness=not_applicable и specificity=not_applicable запрещены,
  в том числе для сравнительного вопроса о названном бренде;
- для unbranded_discovery brand_answer всегда not_applicable;
- общий массив evidence содержит только точные непрерывные фрагменты raw до
  15 слов, без пересказа и без изменения форматирования;
- неизвестность записывай в uncertainties, а не превращай в негатив.

Если research_guidance непустой, используй его только как дополнительное
ужесточение правил текущего исследования. Оно не разрешает расширять каталог,
приписывать бренду новые продукты или менять исходный текст ответа.

{LIVE_RUSSIAN_RULES}
""".strip()
    batch_jobs: list[dict[str, Any]] = []
    batches = _volume_bounded_chunks(
        pending,
        text_key="answer",
        max_items=10,
        max_chars=ANNOTATION_BATCH_CHAR_LIMIT,
    )
    for batch in batches:
        answer_ids = [int(item["answer_id"]) for item in batch]
        digest = hashlib.sha1(
            ",".join(str(value) for value in answer_ids).encode("ascii")
        ).hexdigest()[:12]
        payload = {
            "target": {
                "brand_name": profile.get("brand_name"),
                "aliases": _target_aliases(profile, catalog),
                "site_profile": profile,
            },
            "entity_catalog": catalog.get("entities") or [],
            "entity_attribution_aliases": _entity_attribution_alias_map(
                profile,
                catalog,
            ),
            "research_guidance": research_guidance.strip(),
            "answers": batch,
        }
        batch_jobs.append(
            {
                "batch": batch,
                "answer_ids": answer_ids,
                "artifact_key": f"annotations_{digest}_{ANNOTATION_VERSION}",
                "schema_name": f"aiv_annotations_{digest}",
                "payload": payload,
            }
        )

    semaphore = asyncio.Semaphore(PROCESSING_BATCH_CONCURRENCY)

    async def request_batch(
        job: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, Exception | None]:
        try:
            async with semaphore:
                result = await _processing_artifact(
                    run_id,
                    stage_key="knowledge_gap",
                    artifact_key=str(job["artifact_key"]),
                    schema=ANNOTATION_SCHEMA,
                    schema_name=str(job["schema_name"]),
                    system=system,
                    user_payload=job["payload"],
                    max_tokens=12_000,
                    prompt_version=ANNOTATION_VERSION,
                )
            return result, None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return None, exc

    tasks = [
        asyncio.create_task(
            request_batch(job),
            name=f"aiv-annotation-{index}",
        )
        for index, job in enumerate(batch_jobs, start=1)
    ]
    processed = 0
    try:
        # LLM requests run concurrently, but their results are reconciled,
        # persisted and reflected in progress strictly in batch order.
        for job, task in zip(batch_jobs, tasks, strict=True):
            result, error = await task
            batch = job["batch"]
            answer_ids = job["answer_ids"]
            if error is not None:
                logger.error(
                    "Annotation batch failed for run %s",
                    run_id,
                    exc_info=(type(error), error, error.__traceback__),
                )
            elif result is not None:
                pending_by_id = {
                    int(item["answer_id"]): item
                    for item in batch
                }
                reconciled = [
                    _reconcile_annotation(
                        item,
                        pending_by_id[int(item["answer_id"])],
                        profile,
                        catalog,
                        annotation_input_sha256=annotation_input_sha256,
                    )
                    for item in result.get("answers") or []
                    if isinstance(item, dict)
                    and isinstance(item.get("answer_id"), int)
                    and int(item["answer_id"]) in pending_by_id
                ]
                await _save_annotations(
                    run_id,
                    reconciled,
                    set(answer_ids),
                )
            processed += len(batch)
            ratio = processed / total
            await update_progress(
                run_id,
                stage="knowledge_gap",
                percent=73 + round(ratio * 7),
                detail=(
                    "Размечаем упоминания, позиции и контекст: "
                    f"{round(ratio * 100)}%."
                ),
                eta_seconds=max(180, int(420 * (1 - ratio))),
            )
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async with SessionLocal() as session:
        completed_rows = (
            await session.execute(
                select(ModelAnswer, AnswerAnnotation)
                .outerjoin(
                    AnswerAnnotation,
                    AnswerAnnotation.answer_id == ModelAnswer.id,
                )
                .where(
                    ModelAnswer.run_id == run_id,
                    ModelAnswer.status == "completed",
                )
            )
        ).all()
    eligible_rows = [
        (answer, annotation)
        for answer, annotation in completed_rows
        if str(answer.response_text or "").strip()
    ]
    current_annotations = sum(
        annotation is not None
        and _annotation_matches_answer(
            annotation.annotation_json or {},
            answer_text=str(answer.response_text or ""),
            answer_model=answer.model,
            annotation_input_sha256=annotation_input_sha256,
        )
        for answer, annotation in eligible_rows
    )
    if (
        eligible_rows
        and current_annotations != len(eligible_rows)
        and _completion_attempt < ANNOTATION_COMPLETION_ATTEMPTS
    ):
        await _annotate_answers(
            run_id,
            profile,
            catalog,
            research_guidance=research_guidance,
            _completion_attempt=_completion_attempt + 1,
        )
        return
    if not eligible_rows or current_annotations != len(eligible_rows):
        raise OpenRouterError(
            "Every completed answer must have a current annotation before metrics"
        )


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator * 100, 1) if denominator else None


def _is_portfolio_target_entity(entity: dict[str, Any]) -> bool:
    return bool(
        _catalog_marks_portfolio_entity(entity)
        and entity.get("_profile_membership_confirmed") is True
    )


def _row_has_completed_raw(row: dict[str, Any]) -> bool:
    # Persisted metric rows always carry answer_text. Keeping omitted text
    # compatible here makes the pure metric helper usable with synthetic rows;
    # an explicitly empty persisted value still fails closed.
    has_nonempty_raw = (
        "answer_text" not in row
        or bool(str(row.get("answer_text") or "").strip())
    )
    return bool(
        row.get("status", "completed") == "completed"
        and has_nonempty_raw
        and row.get("metric_eligible", True) is not False
    )


def _slice_evidence_state(
    valid_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    observational = sum(
        row.get("metric_evidence_state") == "legacy_observational"
        for row in valid_rows
    )
    strictly_attested = sum(
        row.get("metric_evidence_state", "strict_verified")
        == "strict_verified"
        for row in valid_rows
    )
    legacy_retrieval_confirmed = sum(
        row.get("metric_evidence_state") == "legacy_retrieval_confirmed"
        for row in valid_rows
    )
    other = max(
        0,
        len(valid_rows)
        - observational
        - strictly_attested
        - legacy_retrieval_confirmed,
    )
    evidence_kinds = sum(
        bool(count)
        for count in (
            observational,
            strictly_attested,
            legacy_retrieval_confirmed,
            other,
        )
    )
    return {
        "strictly_attested_answers": strictly_attested,
        "legacy_retrieval_confirmed_answers": legacy_retrieval_confirmed,
        "observational_answers": observational,
        "evidence_state": (
            "mixed"
            if evidence_kinds > 1
            else "legacy_observational"
            if observational
            else "legacy_retrieval_confirmed"
            if legacy_retrieval_confirmed
            else "attested"
            if valid_rows
            else "unavailable"
        ),
        "limitation_reason": (
            LEGACY_MEMORY_OBSERVATION_REASON if observational else None
        ),
    }
def _visibility_slice(
    rows: list[dict[str, Any]],
    *,
    mode: str,
    provider: str | None = None,
    intent: str | None = None,
    scope: str = "parent",
    entity_catalog: dict[str, dict[str, Any]] | None = None,
    target_aliases: list[str] | None = None,
    scope_available: bool = True,
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["mode"] == mode
        and row["role"] == "unbranded_discovery"
        and (provider is None or row["provider_key"] == provider)
        and (intent is None or row["intent_class"] == intent)
    ]
    expected = len(selected)
    completed = sum(_row_has_completed_raw(row) for row in selected)
    annotated_rows = [
        row
        for row in selected
        if _row_has_completed_raw(row)
        and isinstance(row.get("annotation"), dict)
        and row["annotation"]
    ]
    annotated = len(annotated_rows)
    valid_rows = [
        row
        for row in annotated_rows
        if row["annotation"].get("valid") is True
    ]
    valid = len(valid_rows)
    evidence_state = _slice_evidence_state(valid_rows)
    evidence_state["strict_no_web_verified"] = bool(
        mode == "memory"
        and valid
        and evidence_state["strictly_attested_answers"] == valid
    )

    if scope == "portfolio" and not scope_available:
        return {
            "score": None,
            "mention_rate": None,
            "mention_count": None,
            "top3_rate": None,
            "top3_count": None,
            "recommendation_rate": None,
            "recommendation_count": None,
            "conditional_rate": None,
            "conditional_count": None,
            "expected_answers": expected,
            "completed_answers": completed,
            "annotated_answers": annotated,
            "valid_answers": valid,
            "coverage_rate": (
                _rate(valid, expected)
                if completed or annotated
                else None
            ),
            "data_state": "unavailable",
            "state": "unknown",
            "mentioned_entities": [],
            "unavailable_reason": "target_portfolio_unconfirmed",
            **evidence_state,
        }

    def outcome(row: dict[str, Any]) -> tuple[bool, int | None, str]:
        annotation = row["annotation"]
        mentioned = bool(annotation.get("target_mentioned"))
        position = (
            int(annotation["target_position"])
            if isinstance(annotation.get("target_position"), int)
            else None
        )
        role = str(annotation.get("target_role") or "absent")
        if scope != "portfolio":
            return mentioned, position, role

        mentioned = False
        position = None
        role = "absent"
        role_priority = {
            "absent": 0,
            "unknown": 0,
            "excluded": 1,
            "mentioned": 2,
            "conditional": 3,
            "recommended": 4,
        }
        for mention in annotation.get("entity_mentions") or []:
            if not isinstance(mention, dict):
                continue
            canonical = str(mention.get("canonical_name") or "").casefold()
            entity = (entity_catalog or {}).get(canonical) or {}
            if (
                not _is_portfolio_target_entity(entity)
                or not _portfolio_entity_is_grounded(
                    str(row.get("answer_text") or ""),
                    entity,
                    target_aliases or [],
                    mention,
                )
            ):
                continue
            mentioned = True
            mention_position = mention.get("position")
            if isinstance(mention_position, int):
                position = (
                    mention_position
                    if position is None
                    else min(position, mention_position)
                )
            mention_role = str(mention.get("role") or "mentioned")
            if role_priority.get(mention_role, 0) > role_priority.get(role, 0):
                role = mention_role
        return mentioned, position, role

    outcomes = [outcome(row) for row in valid_rows]
    denominator = valid
    mentions = sum(mentioned for mentioned, _position, _role in outcomes)
    top3 = sum(
        mentioned and isinstance(position, int) and position <= 3
        for mentioned, position, _role in outcomes
    )
    recommended = sum(
        role == "recommended" for _mentioned, _position, role in outcomes
    )
    conditional = sum(
        role == "conditional" for _mentioned, _position, role in outcomes
    )
    mention_rate = _rate(mentions, denominator)
    top3_rate = _rate(top3, denominator)
    recommendation_rate = _rate(recommended, denominator)
    score = (
        round(
            0.25 * mention_rate
            + 0.35 * top3_rate
            + 0.40 * recommendation_rate,
            1,
        )
        if None not in (mention_rate, top3_rate, recommendation_rate)
        else None
    )
    mentioned_entities: Counter[str] = Counter()
    if scope == "portfolio":
        for row in valid_rows:
            answer_entities: set[str] = set()
            for mention in row["annotation"].get("entity_mentions") or []:
                if not isinstance(mention, dict):
                    continue
                canonical = str(mention.get("canonical_name") or "").casefold()
                entity = (entity_catalog or {}).get(canonical) or {}
                if (
                    not _is_portfolio_target_entity(entity)
                    or not _portfolio_entity_is_grounded(
                        str(row.get("answer_text") or ""),
                        entity,
                        target_aliases or [],
                        mention,
                    )
                ):
                    continue
                display_name = str(
                    entity.get("canonical_name")
                    or mention.get("canonical_name")
                    or ""
                ).strip()
                if display_name:
                    answer_entities.add(display_name)
            mentioned_entities.update(answer_entities)
    return {
        "score": score,
        "mention_rate": mention_rate,
        "mention_count": mentions if denominator else None,
        "top3_rate": top3_rate,
        "top3_count": top3 if denominator else None,
        "recommendation_rate": recommendation_rate,
        "recommendation_count": recommended if denominator else None,
        "conditional_rate": _rate(conditional, denominator),
        "conditional_count": conditional if denominator else None,
        "expected_answers": expected,
        "completed_answers": completed,
        "annotated_answers": annotated,
        "valid_answers": valid,
        "coverage_rate": (
            _rate(valid, expected)
            if completed or annotated
            else None
        ),
        "data_state": (
            "complete"
            if expected
            and completed == expected
            and valid == expected
            and not evidence_state["observational_answers"]
            else "limited"
            if completed or annotated
            else "unavailable"
        ),
        "state": (
            "strong"
            if score is not None and score >= 70
            else "visible"
            if score is not None and score >= 40
            else "weak"
            if score is not None
            else "unknown"
        ),
        "mentioned_entities": [
            {
                "name": name,
                "answer_count": count,
                "answer_rate": _rate(count, denominator),
            }
            for name, count in sorted(
                mentioned_entities.items(),
                key=lambda item: (-item[1], item[0].casefold()),
            )
        ],
        **evidence_state,
    }


async def _metric_rows(
    run_id: str,
    *,
    annotation_input_sha256: str,
) -> list[dict[str, Any]]:
    legacy_contract = await _legacy_panel_run_contract(run_id)
    async with SessionLocal() as session:
        rows = (
            await session.execute(
                select(ModelAnswer, VisibilityPrompt, AnswerAnnotation)
                .join(VisibilityPrompt, ModelAnswer.prompt_id == VisibilityPrompt.id)
                .outerjoin(
                    AnswerAnnotation,
                    AnswerAnnotation.answer_id == ModelAnswer.id,
                )
                .where(ModelAnswer.run_id == run_id)
            )
        ).all()
    output: list[dict[str, Any]] = []
    for answer, prompt, annotation in rows:
        answer_text = answer.response_text or ""
        stored = annotation.annotation_json if annotation is not None else {}
        web_attested, web_attestation_reason = _panel_answer_attestation(
            answer,
            prompt_text=prompt.text,
            legacy_allowed=legacy_contract["eligible"],
        )
        annotation_is_current = bool(
            answer.status == "completed"
            and answer_text.strip()
            and _annotation_matches_answer(
                stored or {},
                answer_text=answer_text,
                answer_model=answer.model,
                annotation_input_sha256=annotation_input_sha256,
            )
        )
        metric_access = _panel_metric_access(
            answer,
            transport_attested=web_attested,
            attestation_reason=web_attestation_reason,
            legacy_memory_observation_allowed=legacy_contract.get(
                "memory_observation_eligible",
                False,
            ),
        )
        metric_eligible = bool(
            answer.status == "completed"
            and answer_text.strip()
            and metric_access["metric_eligible"]
        )
        context_eligible = bool(
            answer.status == "completed"
            and answer_text.strip()
            and metric_access["context_eligible"]
        )
        output.append({
            "answer_id": answer.id,
            "mode": answer.mode,
            "provider_key": answer.provider_key,
            "prompt_id": prompt.id,
            "prompt_key": prompt.prompt_key,
            "scenario": prompt.text,
            "intent_class": prompt.intent_class,
            "role": prompt.role,
            "status": answer.status,
            "model": answer.model,
            "citations_count": len(answer.citations_json or []),
            "answer_text": answer_text,
            "web_attested": web_attested,
            "web_attestation_reason": web_attestation_reason,
            "panel_evidence_version": (
                LEGACY_PANEL_EVIDENCE_VERSION
                if web_attestation_reason.startswith("legacy_")
                else WEB_ATTESTATION_VERSION
            ),
            "panel_evidence_sha256": _panel_evidence_sha256(
                answer,
                reason=web_attestation_reason,
                legacy_contract=legacy_contract,
            ),
            "metric_eligible": metric_eligible,
            "context_eligible": context_eligible,
            "metric_evidence_state": metric_access[
                "metric_evidence_state"
            ],
            "metric_limitation": metric_access["metric_limitation"],
            "annotation": stored if annotation_is_current else {},
            "annotation_state": (
                "current"
                if annotation_is_current
                else "ineligible"
                if (
                    answer.status != "completed"
                    or not answer_text.strip()
                )
                else "missing_or_stale"
            ),
        })
    return output


async def _expected_corpus_cells(
    run_id: str,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build the full prompt × provider × mode contract for this run.

    Existing answer models are authoritative for saved historical runs. A
    missing answer row still gets an expected cell from the current panel
    contract, so absence cannot disappear from the manifest.
    """

    async with SessionLocal() as session:
        prompts = list(
            (
                await session.execute(
                    select(VisibilityPrompt)
                    .where(VisibilityPrompt.run_id == run_id)
                    .order_by(VisibilityPrompt.sequence, VisibilityPrompt.id)
                )
            )
            .scalars()
            .all()
        )
    persisted_models = {
        (
            int(row.get("prompt_id") or 0),
            str(row.get("provider_key") or ""),
            str(row.get("mode") or ""),
        ): str(row.get("model") or "")
        for row in rows
    }
    cells: list[dict[str, Any]] = []
    for prompt in prompts:
        for mode in ("web", "memory"):
            for panel in panel_models():
                configured_model = (
                    panel.model if mode == "web" else panel.memory_model
                )
                if configured_model is None:
                    continue
                model = persisted_models.get(
                    (prompt.id, panel.key, mode),
                    configured_model,
                )
                cells.append(
                    {
                        "prompt_id": prompt.id,
                        "provider_key": panel.key,
                        "model": model,
                        "mode": mode,
                    }
                )
    return sorted(cells, key=_corpus_cell_key)


def _consistency_index(
    rows: list[dict[str, Any]],
    *,
    scope: str = "parent",
    entity_catalog: dict[str, dict[str, Any]] | None = None,
    target_aliases: list[str] | None = None,
) -> float | None:
    by_prompt: dict[int, list[int]] = defaultdict(list)
    for row in rows:
        annotation = row["annotation"]
        if (
            row["mode"] == "web"
            and row["role"] == "unbranded_discovery"
            and _row_has_completed_raw(row)
            and annotation.get("valid") is True
        ):
            if scope == "portfolio":
                mentioned = False
                for mention in annotation.get("entity_mentions") or []:
                    canonical = str(
                        (mention or {}).get("canonical_name") or ""
                    ).casefold()
                    entity = (entity_catalog or {}).get(canonical) or {}
                    if _is_portfolio_target_entity(
                        entity
                    ) and _portfolio_entity_is_grounded(
                        str(row.get("answer_text") or ""),
                        entity,
                        target_aliases or [],
                        mention if isinstance(mention, dict) else None,
                    ):
                        mentioned = True
                        break
            else:
                mentioned = bool(annotation.get("target_mentioned"))
            by_prompt[row["prompt_id"]].append(int(mentioned))
    values: list[float] = []
    for outcomes in by_prompt.values():
        if len(outcomes) < 2:
            continue
        probability = sum(outcomes) / len(outcomes)
        if probability in {0, 1}:
            values.append(1.0)
        else:
            entropy = -(
                probability * math.log2(probability)
                + (1 - probability) * math.log2(1 - probability)
            )
            values.append(1 - entropy)
    return round(sum(values) / len(values) * 100, 1) if values else None


def _brand_knowledge_slice(
    rows: list[dict[str, Any]],
    *,
    mode: str,
    provider: str | None = None,
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["mode"] == mode
        and row["role"] == "brand_diagnostic"
        and (provider is None or row["provider_key"] == provider)
    ]
    expected = len(selected)
    completed = sum(_row_has_completed_raw(row) for row in selected)
    annotated_rows = [
        row
        for row in selected
        if _row_has_completed_raw(row)
        and isinstance(row.get("annotation"), dict)
        and row["annotation"]
    ]
    annotated = len(annotated_rows)
    valid_rows = [
        row
        for row in annotated_rows
        if row["annotation"].get("valid") is True
    ]
    valid = len(valid_rows)
    evidence_state = _slice_evidence_state(valid_rows)
    evidence_state["strict_no_web_verified"] = bool(
        mode == "memory"
        and valid
        and evidence_state["strictly_attested_answers"] == valid
    )
    denominator = valid
    answer_count = 0
    specific_count = 0
    contradiction_count = 0
    citation_count = 0
    facets: Counter[str] = Counter()
    for row in valid_rows:
        brand_answer = row["annotation"].get("brand_answer")
        if not isinstance(brand_answer, dict):
            continue
        specificity = str(brand_answer.get("specificity") or "none")
        directness = str(brand_answer.get("directness") or "refusal")
        if (
            directness in {"direct", "partial"}
            and specificity in {"specific", "generic", "contradictory"}
        ):
            answer_count += 1
        if specificity == "specific":
            specific_count += 1
        if specificity == "contradictory" or brand_answer.get("contradictions"):
            contradiction_count += 1
        if int(row.get("citations_count") or 0) > 0:
            citation_count += 1
        for facet in set(brand_answer.get("supported_facets") or []):
            facets[str(facet)] += 1
    specific_rate = _rate(specific_count, denominator)
    return {
        "answer_rate": _rate(answer_count, denominator),
        "answer_count": answer_count if denominator else None,
        "specific_rate": specific_rate,
        "specific_count": specific_count if denominator else None,
        "contradiction_rate": _rate(contradiction_count, denominator),
        "contradiction_count": contradiction_count if denominator else None,
        "citation_rate": _rate(citation_count, denominator),
        "citation_count": citation_count if denominator else None,
        "expected_answers": expected,
        "completed_answers": completed,
        "annotated_answers": annotated,
        "valid_answers": valid,
        "coverage_rate": (
            _rate(valid, expected)
            if completed or annotated
            else None
        ),
        "data_state": (
            "complete"
            if expected
            and completed == expected
            and valid == expected
            and not evidence_state["observational_answers"]
            else "limited"
            if completed or annotated
            else "unavailable"
        ),
        "state": (
            "strong"
            if specific_rate is not None and specific_rate >= 70
            else "visible"
            if specific_rate is not None and specific_rate >= 40
            else "weak"
            if specific_rate is not None
            else "unknown"
        ),
        "facets": [
            {
                "name": name,
                "answer_count": count,
                "answer_rate": _rate(count, denominator),
            }
            for name, count in sorted(
                facets.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ],
        **evidence_state,
    }


def _compute_metrics(
    rows: list[dict[str, Any]],
    profile: dict[str, Any],
    catalog: dict[str, Any],
) -> dict[str, Any]:
    # Re-apply the deterministic scope guard here as well. This keeps metrics
    # correct for cached historical catalogs created before canonical-name
    # collision resolution was introduced.
    candidate_portfolio_count = sum(
        _catalog_marks_portfolio_entity(entity)
        for entity in catalog.get("entities") or []
        if isinstance(entity, dict)
    )
    scoped_catalog = _scope_entity_catalog_to_profile(catalog, profile)
    target_aliases = _target_aliases(profile, scoped_catalog)
    confirmed_owner_aliases = _attribution_owner_aliases(
        profile,
        scoped_catalog,
    )
    entity_catalog: dict[str, dict[str, Any]] = {}
    for entity in scoped_catalog.get("entities") or []:
        if not isinstance(entity, dict) or not entity.get("canonical_name"):
            continue
        normalized = dict(entity)
        if _catalog_marks_portfolio_entity(normalized):
            normalized["_profile_membership_confirmed"] = (
                _profile_confirms_portfolio_entity(normalized, profile)
            )
            normalized["_mention_policy"] = _portfolio_mention_policy(
                normalized,
                profile,
            )
            normalized["_attribution_aliases"] = (
                _entity_attribution_aliases(
                    profile,
                    scoped_catalog,
                    normalized,
                )
            )
            normalized["_direct_target_aliases"] = target_aliases
            normalized["_confirmed_owner_aliases"] = (
                confirmed_owner_aliases
            )
        entity_catalog[
            str(normalized.get("canonical_name") or "").casefold()
        ] = normalized
    confirmed_portfolio_entities = [
        entity
        for entity in entity_catalog.values()
        if _is_portfolio_target_entity(entity)
    ]
    confirmed_portfolio_count = len(confirmed_portfolio_entities)
    portfolio_scope_available = confirmed_portfolio_count > 0
    portfolio_scope_state = (
        "complete"
        if confirmed_portfolio_count
        and confirmed_portfolio_count == candidate_portfolio_count
        else "limited"
        if confirmed_portfolio_count
        else "unavailable"
    )
    parent_web = _visibility_slice(
        rows,
        mode="web",
        scope="parent",
        entity_catalog=entity_catalog,
        target_aliases=target_aliases,
    )
    parent_memory = _visibility_slice(
        rows,
        mode="memory",
        scope="parent",
        entity_catalog=entity_catalog,
        target_aliases=target_aliases,
    )
    portfolio_web = _visibility_slice(
        rows,
        mode="web",
        scope="portfolio",
        entity_catalog=entity_catalog,
        target_aliases=target_aliases,
        scope_available=portfolio_scope_available,
    )
    portfolio_memory = _visibility_slice(
        rows,
        mode="memory",
        scope="portfolio",
        entity_catalog=entity_catalog,
        target_aliases=target_aliases,
        scope_available=portfolio_scope_available,
    )
    brand_web = _brand_knowledge_slice(rows, mode="web")
    brand_memory = _brand_knowledge_slice(rows, mode="memory")
    labels = {model.key: model.label for model in panel_models()}
    providers = [
        {
            "name": labels.get(model.key, model.key),
            "parent_discovery": _visibility_slice(
                rows,
                mode="web",
                provider=model.key,
                scope="parent",
                entity_catalog=entity_catalog,
                target_aliases=target_aliases,
            ),
            "portfolio_capture": _visibility_slice(
                rows,
                mode="web",
                provider=model.key,
                scope="portfolio",
                entity_catalog=entity_catalog,
                target_aliases=target_aliases,
                scope_available=portfolio_scope_available,
            ),
            "brand_knowledge": {
                "web": _brand_knowledge_slice(
                    rows,
                    mode="web",
                    provider=model.key,
                ),
                "memory": _brand_knowledge_slice(
                    rows,
                    mode="memory",
                    provider=model.key,
                ),
            },
        }
        for model in panel_models()
    ]
    intents = [
        {
            "intent": intent,
            "parent_discovery": _visibility_slice(
                rows,
                mode="web",
                intent=intent,
                scope="parent",
                entity_catalog=entity_catalog,
                target_aliases=target_aliases,
            ),
            "portfolio_capture": _visibility_slice(
                rows,
                mode="web",
                intent=intent,
                scope="portfolio",
                entity_catalog=entity_catalog,
                target_aliases=target_aliases,
                scope_available=portfolio_scope_available,
            ),
        }
        for intent in ("I", "E", "T", "NB", "NAV", "TR")
    ]

    target_name = str(profile.get("brand_name") or "Целевой бренд")
    events: Counter[str] = Counter()
    portfolio_events: Counter[str] = Counter()
    answer_denominator = 0
    for row in rows:
        annotation = row["annotation"]
        if (
            row["mode"] != "web"
            or row["role"] != "unbranded_discovery"
            or not _row_has_completed_raw(row)
            or annotation.get("valid") is not True
        ):
            continue
        answer_denominator += 1
        if annotation.get("target_mentioned"):
            events[target_name] += 1
        seen_entity_keys: set[str] = set()
        for mention in annotation.get("entity_mentions") or []:
            canonical = str(mention.get("canonical_name") or "").strip()
            entity = entity_catalog.get(canonical.casefold()) or {}
            normalized_name = str(
                entity.get("canonical_name") or canonical
            ).strip()
            normalized_key = normalized_name.casefold()
            if (
                normalized_name
                and normalized_key not in seen_entity_keys
                and entity.get("category") in {"target", "competitor"}
            ):
                seen_entity_keys.add(normalized_key)
                if _is_portfolio_target_entity(
                    entity
                ) and _portfolio_entity_is_grounded(
                    str(row.get("answer_text") or ""),
                    entity,
                    target_aliases,
                    mention,
                ):
                    portfolio_events[normalized_name] += 1
                elif entity.get("category") == "competitor":
                    events[normalized_name] += 1
    target_count = events.pop(target_name, 0)
    competitors = (
        [
            {
                "name": target_name,
                "mention_count": target_count,
                "total_answers": answer_denominator,
                "mention_share": _rate(target_count, answer_denominator),
                "is_target": True,
                "relationship": "parent",
            },
            *[
                {
                    "name": name,
                    "mention_count": count,
                    "total_answers": answer_denominator,
                    "mention_share": _rate(count, answer_denominator),
                    "is_target": True,
                    "relationship": "portfolio",
                }
                for name, count in portfolio_events.most_common(4)
            ],
            *[
                {
                    "name": name,
                    "mention_count": count,
                    "total_answers": answer_denominator,
                    "mention_share": _rate(count, answer_denominator),
                    "is_target": False,
                    "relationship": "competitor",
                }
                for name, count in events.most_common(8)
            ],
        ][:12]
        if answer_denominator
        else []
    )

    sentiments = Counter()
    for row in rows:
        annotation = row["annotation"]
        if (
            row["mode"] == "web"
            and row["role"] == "unbranded_discovery"
            and _row_has_completed_raw(row)
            and annotation.get("valid") is True
            and annotation.get("target_mentioned")
            and annotation.get("sentiment") != "unknown"
        ):
            sentiments[str(annotation.get("sentiment"))] += 1
    known_sentiments = sum(sentiments.values())
    sentiment = {
        key: _rate(value, known_sentiments)
        for key, value in sentiments.items()
    }

    web_keys = {
        (row["provider_key"], row["prompt_id"])
        for row in rows
        if row["mode"] == "web"
        and row["role"] == "unbranded_discovery"
        and _row_has_completed_raw(row)
        and isinstance(row.get("annotation"), dict)
        and row["annotation"].get("valid") is True
    }
    memory_keys = {
        (row["provider_key"], row["prompt_id"])
        for row in rows
        if row["mode"] == "memory"
        and row["role"] == "unbranded_discovery"
        and _row_has_completed_raw(row)
        and isinstance(row.get("annotation"), dict)
        and row["annotation"].get("valid") is True
    }
    paired_keys = web_keys & memory_keys
    paired_rows = [
        row
        for row in rows
        if (row["provider_key"], row["prompt_id"]) in paired_keys
    ]
    paired_parent_web = _visibility_slice(
        paired_rows,
        mode="web",
        scope="parent",
        entity_catalog=entity_catalog,
        target_aliases=target_aliases,
    )
    paired_parent_memory = _visibility_slice(
        paired_rows,
        mode="memory",
        scope="parent",
        entity_catalog=entity_catalog,
        target_aliases=target_aliases,
    )
    paired_portfolio_web = _visibility_slice(
        paired_rows,
        mode="web",
        scope="portfolio",
        entity_catalog=entity_catalog,
        target_aliases=target_aliases,
        scope_available=portfolio_scope_available,
    )
    paired_portfolio_memory = _visibility_slice(
        paired_rows,
        mode="memory",
        scope="portfolio",
        entity_catalog=entity_catalog,
        target_aliases=target_aliases,
        scope_available=portfolio_scope_available,
    )

    def score_lift(
        web_slice: dict[str, Any],
        memory_slice: dict[str, Any],
    ) -> float | None:
        if web_slice.get("score") is None or memory_slice.get("score") is None:
            return None
        return round(
            float(web_slice["score"]) - float(memory_slice["score"]),
            1,
        )

    def mention_rate_difference(
        web_slice: dict[str, Any],
        memory_slice: dict[str, Any],
    ) -> float | None:
        """Return the descriptive difference shown beside mention-rate cells."""

        web_rate = web_slice.get("mention_rate")
        memory_rate = memory_slice.get("mention_rate")
        if web_rate is None or memory_rate is None:
            return None
        return round(float(web_rate) - float(memory_rate), 1)

    parent_lift = score_lift(paired_parent_web, paired_parent_memory)
    portfolio_lift = score_lift(
        paired_portfolio_web,
        paired_portfolio_memory,
    )
    paired_observational = bool(
        paired_parent_memory.get("observational_answers")
        or paired_portfolio_memory.get("observational_answers")
    )
    parent_observed_difference = mention_rate_difference(
        paired_parent_web,
        paired_parent_memory,
    )
    portfolio_observed_difference = mention_rate_difference(
        paired_portfolio_web,
        paired_portfolio_memory,
    )
    if paired_observational:
        parent_lift = None
        portfolio_lift = None
    raw_completed_answers = sum(
        row.get("status", "completed") == "completed"
        and (
            "answer_text" not in row
            or bool(str(row.get("answer_text") or "").strip())
        )
        for row in rows
    )
    completed_annotations = sum(
        row.get("status", "completed") == "completed"
        and (
            "answer_text" not in row
            or bool(str(row.get("answer_text") or "").strip())
        )
        and bool(row.get("annotation"))
        for row in rows
    )
    metric_eligible_answers = sum(_row_has_completed_raw(row) for row in rows)
    observational_metric_answers = sum(
        _row_has_completed_raw(row)
        and row.get("metric_evidence_state") == "legacy_observational"
        for row in rows
    )
    context_eligible_answers = sum(
        row.get("status", "completed") == "completed"
        and (
            "answer_text" not in row
            or bool(str(row.get("answer_text") or "").strip())
        )
        and row.get(
            "context_eligible",
            row.get("metric_eligible", True),
        ) is not False
        for row in rows
    )
    valid_annotations = sum(
        row["annotation"].get("valid") is True
        for row in rows
        if _row_has_completed_raw(row) and row.get("annotation")
    )
    expected_answers = len(rows)
    evidence_reasons = Counter(
        str(row.get("web_attestation_reason") or "unknown")
        for row in rows
    )
    quality_state = (
        "good"
        if expected_answers
        and raw_completed_answers == expected_answers
        and completed_annotations == expected_answers
        and metric_eligible_answers == expected_answers
        and valid_annotations == expected_answers
        and observational_metric_answers == 0
        else "limited"
        if raw_completed_answers or completed_annotations
        else "unavailable"
    )
    return {
        "parent_discovery": {
            "web": parent_web,
            "memory": parent_memory,
        },
        "portfolio_visibility": {
            "web": portfolio_web,
            "memory": portfolio_memory,
        },
        "portfolio_scope": {
            "state": portfolio_scope_state,
            "candidate_entities": candidate_portfolio_count,
            "confirmed_entities": confirmed_portfolio_count,
            "rejected_entities": max(
                0,
                candidate_portfolio_count - confirmed_portfolio_count,
            ),
            "reason": (
                None
                if portfolio_scope_available
                else "target_portfolio_unconfirmed"
            ),
        },
        "brand_knowledge": {
            "web": brand_web,
            "memory": brand_memory,
            "providers": [
                {
                    "name": item["name"],
                    **item["brand_knowledge"],
                }
                for item in providers
            ],
        },
        "paired_web_lift": {
            "n_pairs": len(paired_keys),
            "data_state": (
                "limited"
                if paired_observational
                else "complete"
                if paired_keys
                else "unavailable"
            ),
            "limitation_reason": (
                LEGACY_MEMORY_OBSERVATION_REASON
                if paired_observational
                else None
            ),
            "causal_interpretation_allowed": False,
            "parent": {
                "web": paired_parent_web,
                "memory": paired_parent_memory,
                "score_lift": parent_lift,
                "observed_difference": parent_observed_difference,
                "observed_difference_metric": "mention_rate_percentage_points",
            },
            "portfolio": {
                "web": paired_portfolio_web,
                "memory": paired_portfolio_memory,
                "score_lift": portfolio_lift,
                "observed_difference": portfolio_observed_difference,
                "observed_difference_metric": "mention_rate_percentage_points",
                "data_state": (
                    "complete"
                    if portfolio_scope_available
                    and paired_portfolio_web.get("data_state") == "complete"
                    and paired_portfolio_memory.get("data_state") == "complete"
                    else "limited"
                    if portfolio_scope_available
                    else "unavailable"
                ),
                "state": (
                    "available"
                    if portfolio_scope_available
                    and paired_portfolio_web.get("data_state") == "complete"
                    and paired_portfolio_memory.get("data_state") == "complete"
                    else "limited"
                    if portfolio_scope_available
                    else "unknown"
                ),
                "unavailable_reason": (
                    None
                    if portfolio_scope_available
                    else "target_portfolio_unconfirmed"
                ),
            },
        },
        # Backward-compatible aliases use the commercially relevant target
        # scope, while the explicit parent metric remains available above.
        "web": portfolio_web,
        "memory": portfolio_memory,
        "knowledge_gap": portfolio_lift,
        "knowledge_gap_state": (
            "web_lift"
            if portfolio_lift is not None and portfolio_lift >= 10
            else "memory_advantage"
            if portfolio_lift is not None and portfolio_lift <= -10
            else "close"
            if portfolio_lift is not None
            else "unknown"
        ),
        "model_consistency": (
            _consistency_index(
                rows,
                scope="portfolio",
                entity_catalog=entity_catalog,
                target_aliases=target_aliases,
            )
            if portfolio_scope_available
            else None
        ),
        "providers": providers,
        "intents": intents,
        "competitors": competitors,
        "sentiment": sentiment,
        "quality": {
            "state": quality_state,
            "expected_answers": expected_answers,
            "completed_answers": raw_completed_answers,
            "annotated_answers": completed_annotations,
            "metric_eligible_answers": metric_eligible_answers,
            "metric_ineligible_answers": (
                expected_answers - metric_eligible_answers
            ),
            "context_eligible_answers": context_eligible_answers,
            "context_withheld_answers": (
                expected_answers - context_eligible_answers
            ),
            "legacy_observational_answers": observational_metric_answers,
            "valid_answers": valid_annotations,
            "coverage_rate": _rate(valid_annotations, expected_answers),
            "panel_evidence": {
                "strict_verified": evidence_reasons.get("verified", 0),
                "legacy_web_confirmed": evidence_reasons.get(
                    "legacy_web_retrieval_confirmed",
                    0,
                ),
                "legacy_memory_unverified": evidence_reasons.get(
                    LEGACY_MEMORY_OBSERVATION_REASON,
                    0,
                ),
                "legacy_memory_observational": observational_metric_answers,
                "excluded_by_reason": {
                    reason: count
                    for reason, count in sorted(
                        Counter(
                            str(row.get("web_attestation_reason") or "unknown")
                            for row in rows
                            if not _row_has_completed_raw(row)
                        ).items()
                    )
                },
            },
        },
        "metric_note": (
            "Показатели описывают экспресс-снимок выбранных сценариев. "
            "Они помогают найти разрывы, но не заменяют репрезентативное исследование рынка."
        ),
    }


def _stable_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _critic_row_provenance(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    signatures = [
        {
            "answer_id": row.get("answer_id"),
            "mode": row.get("mode"),
            "provider_key": row.get("provider_key"),
            "prompt_id": row.get("prompt_id"),
            "prompt_key": row.get("prompt_key"),
            "scenario": row.get("scenario"),
            "role": row.get("role"),
            "intent_class": row.get("intent_class"),
            "status": row.get("status"),
            "model": row.get("model"),
            "web_attested": row.get("web_attested"),
            "web_attestation_reason": row.get("web_attestation_reason"),
            "panel_evidence_version": row.get("panel_evidence_version"),
            "panel_evidence_sha256": row.get("panel_evidence_sha256"),
            "metric_eligible": row.get("metric_eligible"),
            "context_eligible": row.get("context_eligible"),
            "metric_evidence_state": row.get("metric_evidence_state"),
            "metric_limitation": row.get("metric_limitation"),
            "annotation_state": row.get("annotation_state"),
            "raw_answer_sha256": hashlib.sha256(
                str(row.get("answer_text") or "").encode("utf-8")
            ).hexdigest(),
            "annotation": row.get("annotation") or {},
        }
        for row in rows
    ]
    return sorted(
        signatures,
        key=lambda item: (
            str(item.get("answer_id") or ""),
            str(item.get("mode") or ""),
            str(item.get("provider_key") or ""),
            str(item.get("scenario") or ""),
        ),
    )


def _corpus_cell_key(row: dict[str, Any]) -> tuple[int, str, str, str]:
    return (
        int(row.get("prompt_id") or 0),
        str(row.get("provider_key") or ""),
        str(row.get("model") or ""),
        str(row.get("mode") or ""),
    )


def _expected_corpus_cells_from_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return the exact panel cells represented by persisted answer rows."""

    cells = {
        _corpus_cell_key(row): {
            "prompt_id": int(row.get("prompt_id") or 0),
            "provider_key": str(row.get("provider_key") or ""),
            "model": str(row.get("model") or ""),
            "mode": str(row.get("mode") or ""),
        }
        for row in rows
    }
    return [cells[key] for key in sorted(cells)]


def _final_corpus_manifest(
    rows: list[dict[str, Any]],
    *,
    expected_cells: list[dict[str, Any]],
) -> dict[str, Any]:
    """Describe the exact answer corpus without hiding missing/stale cells."""

    normalized_expected = sorted(
        (
            {
                "prompt_id": int(cell.get("prompt_id") or 0),
                "provider_key": str(cell.get("provider_key") or ""),
                "model": str(cell.get("model") or ""),
                "mode": str(cell.get("mode") or ""),
            }
            for cell in expected_cells
        ),
        key=_corpus_cell_key,
    )
    expected_key_counts = Counter(
        _corpus_cell_key(cell) for cell in normalized_expected
    )
    expected_keys = set(expected_key_counts)
    duplicate_expected_cells = [
        {
            "prompt_id": key[0],
            "provider_key": key[1],
            "model": key[2],
            "mode": key[3],
            "occurrences": count,
        }
        for key, count in sorted(expected_key_counts.items())
        if count > 1
    ]
    observed: list[dict[str, Any]] = []
    observed_by_key: dict[
        tuple[int, str, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)
    invalid_cells: list[dict[str, Any]] = []

    for row in rows:
        annotation = (
            row.get("annotation")
            if isinstance(row.get("annotation"), dict)
            else {}
        )
        raw_answer = str(row.get("answer_text") or "")
        raw_sha256 = hashlib.sha256(raw_answer.encode("utf-8")).hexdigest()
        annotation_input_sha256 = str(
            annotation.get("_annotation_input_sha256") or ""
        )
        annotation_provenance = {
            "annotation_version": annotation.get("_annotation_version"),
            "answer_sha256": annotation.get("_answer_sha256"),
            "answer_model": annotation.get("_answer_model"),
            "annotation_input_sha256": annotation_input_sha256,
        }
        item = {
            "prompt_id": int(row.get("prompt_id") or 0),
            "provider_key": str(row.get("provider_key") or ""),
            "model": str(row.get("model") or ""),
            "mode": str(row.get("mode") or ""),
            "answer_id": row.get("answer_id"),
            "status": str(row.get("status") or ""),
            "raw_answer_sha256": raw_sha256,
            "annotation_sha256": _stable_json_sha256(annotation),
            "annotation_input_sha256": annotation_input_sha256,
            "annotation_provenance_sha256": _stable_json_sha256(
                annotation_provenance
            ),
        }
        observed.append(item)
        observed_by_key[_corpus_cell_key(item)].append(item)

        reasons: list[str] = []
        if not isinstance(item["answer_id"], int):
            reasons.append("answer_id_missing")
        if item["status"] != "completed":
            reasons.append("answer_not_completed")
        if not raw_answer.strip():
            reasons.append("raw_answer_missing")
        if row.get("annotation_state") != "current":
            reasons.append("annotation_missing_or_stale")
        if not annotation:
            reasons.append("annotation_missing")
        if annotation.get("_answer_sha256") != raw_sha256:
            reasons.append("annotation_raw_hash_mismatch")
        if annotation.get("_answer_model") != item["model"]:
            reasons.append("annotation_model_mismatch")
        if not annotation_input_sha256:
            reasons.append("annotation_input_provenance_missing")
        if not isinstance(annotation.get("valid"), bool):
            reasons.append("annotation_validity_missing")
        if reasons:
            invalid_cells.append(
                {
                    **{
                        field: item[field]
                        for field in (
                            "prompt_id",
                            "provider_key",
                            "model",
                            "mode",
                            "answer_id",
                        )
                    },
                    "reasons": reasons,
                }
            )

    observed.sort(
        key=lambda item: (
            *_corpus_cell_key(item),
            int(item.get("answer_id") or 0),
        )
    )
    observed_keys = set(observed_by_key)
    missing_cells = [
        cell
        for cell in normalized_expected
        if _corpus_cell_key(cell) not in observed_keys
    ]
    unexpected_cells = [
        {
            "prompt_id": key[0],
            "provider_key": key[1],
            "model": key[2],
            "mode": key[3],
        }
        for key in sorted(observed_keys - expected_keys)
    ]
    duplicate_cells = [
        {
            "prompt_id": key[0],
            "provider_key": key[1],
            "model": key[2],
            "mode": key[3],
            "answer_ids": sorted(
                int(item["answer_id"])
                for item in items
                if isinstance(item.get("answer_id"), int)
            ),
        }
        for key, items in sorted(observed_by_key.items())
        if len(items) > 1
    ]
    complete = bool(normalized_expected) and not any(
        (
            missing_cells,
            unexpected_cells,
            duplicate_expected_cells,
            duplicate_cells,
            invalid_cells,
        )
    ) and len(observed) == len(normalized_expected)
    digest_input = {
        "version": FINAL_CORPUS_MANIFEST_VERSION,
        "expected_cells": normalized_expected,
        "observed_cells": observed,
        "missing_cells": missing_cells,
        "unexpected_cells": unexpected_cells,
        "duplicate_expected_cells": duplicate_expected_cells,
        "duplicate_cells": duplicate_cells,
        "invalid_cells": invalid_cells,
        "complete": complete,
    }
    return {
        **digest_input,
        "expected_count": len(normalized_expected),
        "observed_count": len(observed),
        "answer_ids": [
            int(item["answer_id"])
            for item in observed
            if isinstance(item.get("answer_id"), int)
        ],
        "expected_cells_sha256": _stable_json_sha256(normalized_expected),
        "observed_cells_sha256": _stable_json_sha256(observed),
        "critic_rows_sha256": _stable_json_sha256(
            _critic_row_provenance(rows)
        ),
        "digest": _stable_json_sha256(digest_input),
    }


def _critic_provenance_digests(
    *,
    profile: dict[str, Any],
    catalog: dict[str, Any],
    rows: list[dict[str, Any]],
    metrics: dict[str, Any],
    policy_history: list[dict[str, Any]],
) -> dict[str, str]:
    """Bind a critic decision to facts, evidence and calculated output."""

    return {
        "version": ANALYSIS_CRITIC_VERSION,
        "profile_sha256": _stable_json_sha256(profile),
        "catalog_sha256": _stable_json_sha256(catalog),
        "rows_sha256": _stable_json_sha256(_critic_row_provenance(rows)),
        "metrics_sha256": _stable_json_sha256(metrics),
        "policy_history_sha256": _stable_json_sha256(policy_history),
    }


def _critic_policy_guidance(policy_history: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        str(step.get("annotation_guidance") or "").strip()
        for step in policy_history
        if isinstance(step, dict)
        and str(step.get("annotation_guidance") or "").strip()
    )


def _deterministic_metric_warnings(
    metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    """Surface suspicious shapes for the critic without declaring them errors."""

    providers = [
        item
        for item in metrics.get("providers") or []
        if isinstance(item, dict)
    ]
    identical_scopes: list[str] = []
    parent_rates: list[float] = []
    portfolio_rates: list[float] = []
    for provider in providers:
        parent = provider.get("parent_discovery") or {}
        portfolio = provider.get("portfolio_capture") or {}
        parent_rate = parent.get("mention_rate")
        portfolio_rate = portfolio.get("mention_rate")
        if isinstance(parent_rate, (int, float)):
            parent_rates.append(float(parent_rate))
        if isinstance(portfolio_rate, (int, float)):
            portfolio_rates.append(float(portfolio_rate))
        if (
            parent.get("valid_answers")
            and parent.get("valid_answers") == portfolio.get("valid_answers")
            and parent.get("mention_count") == portfolio.get("mention_count")
            and parent_rate == portfolio_rate
        ):
            identical_scopes.append(str(provider.get("name") or ""))

    warnings: list[dict[str, Any]] = []
    if len(identical_scopes) >= 3:
        warnings.append(
            {
                "code": "scope_leakage",
                "severity": "important",
                "finding": (
                    "У трёх или более систем совпали числители материнского "
                    "бренда и портфеля; нужно проверить наследование scope."
                ),
                "providers": identical_scopes,
            }
        )
    if len(parent_rates) >= 4 and len(set(parent_rates)) == 1:
        warnings.append(
            {
                "code": "provider_uniformity",
                "severity": "observation",
                "finding": (
                    "Все доступные системы дали одинаковую долю обнаружения "
                    "материнского бренда."
                ),
                "providers": [
                    str(item.get("name") or "") for item in providers
                ],
            }
        )
    if len(portfolio_rates) >= 4 and len(set(portfolio_rates)) == 1:
        warnings.append(
            {
                "code": "provider_uniformity",
                "severity": "observation",
                "finding": (
                    "Все доступные системы дали одинаковую долю обнаружения "
                    "портфеля."
                ),
                "providers": [
                    str(item.get("name") or "") for item in providers
                ],
            }
        )
    return warnings


def _deterministic_annotation_warnings(
    *,
    profile: dict[str, Any],
    catalog: dict[str, Any],
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Expose systematic evidence rejection to the independent critic."""

    entities = {
        str(entity.get("canonical_name") or "").casefold(): entity
        for entity in catalog.get("entities") or []
        if isinstance(entity, dict) and entity.get("canonical_name")
    }
    direct_aliases = _target_aliases(profile, catalog)
    confirmed_owner_aliases = _attribution_owner_aliases(profile, catalog)
    rejected: list[dict[str, Any]] = []
    invalid_brand_answers: list[int] = []
    for row in rows:
        annotation = row.get("annotation") or {}
        if (
            row.get("role") == "brand_diagnostic"
            and _row_has_completed_raw(row)
            and annotation.get("valid") is True
        ):
            brand_answer = annotation.get("brand_answer")
            if not isinstance(brand_answer, dict) or (
                brand_answer.get("directness") == "not_applicable"
                or brand_answer.get("specificity") == "not_applicable"
            ):
                answer_id = row.get("answer_id")
                if isinstance(answer_id, int):
                    invalid_brand_answers.append(answer_id)
        if (
            row.get("role") != "unbranded_discovery"
            or not _row_has_completed_raw(row)
            or annotation.get("valid") is not True
        ):
            continue
        for mention in annotation.get("entity_mentions") or []:
            if (
                not isinstance(mention, dict)
                or mention.get("attributed_to_target") is not True
            ):
                continue
            canonical = str(
                mention.get("canonical_name") or ""
            ).casefold()
            entity = entities.get(canonical) or {}
            if (
                not _catalog_marks_portfolio_entity(entity)
                or not _profile_confirms_portfolio_entity(entity, profile)
                or _portfolio_entity_is_grounded(
                    str(row.get("answer_text") or ""),
                    entity,
                    _entity_attribution_aliases(
                        profile,
                        catalog,
                        entity,
                    ),
                    mention,
                    direct_target_aliases=direct_aliases,
                    confirmed_owner_aliases=confirmed_owner_aliases,
                )
            ):
                continue
            rejected.append(
                {
                    "answer_id": row.get("answer_id"),
                    "entity": entity.get("canonical_name"),
                    "evidence": str(mention.get("evidence") or "")[:1000],
                }
            )

    answer_ids = sorted(
        {
            int(item["answer_id"])
            for item in rejected
            if isinstance(item.get("answer_id"), int)
        }
    )
    warnings: list[dict[str, Any]] = []
    if len(answer_ids) >= 3:
        warnings.append(
            {
                "code": "annotation_evidence_mismatch",
                "severity": "important",
                "finding": (
                    "Разметка заявила явную связь подтверждённой услуги "
                    "с целевым брендом, но буквальная проверка evidence "
                    f"отклонила её в {len(answer_ids)} ответах."
                ),
                "answer_ids": answer_ids,
                "entities": sorted(
                    {
                        str(item.get("entity") or "")
                        for item in rejected
                        if item.get("entity")
                    }
                ),
                "rejected_evidence": rejected[:24],
            }
        )
    if invalid_brand_answers:
        warnings.append(
            {
                "code": "brand_knowledge_false_negative",
                "severity": "important",
                "finding": (
                    "В брендовом сценарии brand_answer ошибочно использует "
                    "not_applicable; строку нужно переразметить до публикации."
                ),
                "answer_ids": sorted(set(invalid_brand_answers)),
                "entities": [str(profile.get("brand_name") or "")],
            }
        )
    return warnings


def _critic_payload(
    *,
    profile: dict[str, Any],
    catalog: dict[str, Any],
    rows: list[dict[str, Any]],
    metrics: dict[str, Any],
    policy_history: list[dict[str, Any]],
) -> dict[str, Any]:
    scoped_catalog = _scope_entity_catalog_to_profile(catalog, profile)
    return {
        "site_profile": profile,
        "entity_catalog": scoped_catalog,
        "metric_contract": {
            "parent_discovery": (
                "Только scenario_role=unbranded_discovery; источник события "
                "annotation.target_mentioned."
            ),
            "portfolio_visibility": (
                "Только scenario_role=unbranded_discovery; засчитывается "
                "profile-confirmed target portfolio entity с буквальным "
                "grounding. Для mention_policy=standalone буквального имени "
                "достаточно: attributed_to_target=false корректен, если сам "
                "raw-ответ не называет материнский бренд, и не отменяет "
                "подтверждённую сайтом принадлежность. Для общей услуги с "
                "requires_target_attribution нужна явная связь в raw с "
                "одним из разрешённых для этой сущности алиасов из "
                "entity_attribution_aliases. Общий upstream owner сам по "
                "себе не переносит предложения соседних продуктов. "
                "entity_mentions из brand_diagnostic сюда не входят."
            ),
            "portfolio_scope": (
                "Если site_profile не подтвердил ни одной portfolio-сущности, "
                "портфельный срез unavailable: его значения null, а не ноль. "
                "Ответы панели и answer-derived catalog не могут сами создать "
                "принадлежность."
            ),
            "brand_knowledge": (
                "Только scenario_role=brand_diagnostic; источник события "
                "annotation.brand_answer. Вспомогательные entity_mentions "
                "не меняют этот срез."
            ),
            "missing_data": (
                "Неполный raw или stale/missing annotation исключается из "
                "валидного знаменателя и не становится нулём."
            ),
            "legacy_memory_observation": (
                "metric_evidence_state=legacy_observational допускается в "
                "детерминированных агрегатах только как limited: модель и "
                "отсутствие retrieval-сигналов подтверждены, но явное "
                "отключение веб-доступа не сохранено. context_eligible=false "
                "запрещает передавать raw финальному автору. Такой срез не "
                "доказывает причинный эффект веб-поиска."
            ),
        },
        "attribution_owner_aliases": _attribution_owner_aliases(
            profile,
            scoped_catalog,
        ),
        "entity_attribution_aliases": _entity_attribution_alias_map(
            profile,
            scoped_catalog,
        ),
        "candidate_metrics": metrics,
        "deterministic_warnings": [
            *_deterministic_metric_warnings(metrics),
            *_deterministic_annotation_warnings(
                profile=profile,
                catalog=scoped_catalog,
                rows=rows,
            ),
        ],
        "previous_policy_changes": policy_history,
        "answers": [
            {
                "answer_id": row.get("answer_id"),
                "mode": row.get("mode"),
                "provider": row.get("provider_key"),
                "model": row.get("model"),
                "prompt_id": row.get("prompt_id"),
                "prompt_key": row.get("prompt_key"),
                "scenario": row.get("scenario"),
                "scenario_role": row.get("role"),
                "intent_class": row.get("intent_class"),
                "status": row.get("status"),
                "annotation_state": row.get("annotation_state"),
                "metric_eligible": row.get("metric_eligible"),
                "context_eligible": row.get("context_eligible"),
                "metric_evidence_state": row.get(
                    "metric_evidence_state"
                ),
                "metric_limitation": row.get("metric_limitation"),
                "panel_evidence_version": row.get("panel_evidence_version"),
                "panel_evidence_reason": row.get("web_attestation_reason"),
                "panel_evidence_sha256": row.get("panel_evidence_sha256"),
                "citations_count": row.get("citations_count"),
                "raw_answer_sha256": hashlib.sha256(
                    str(row.get("answer_text") or "").encode("utf-8")
                ).hexdigest(),
                "raw_answer_truncated": (
                    len(str(row.get("answer_text") or ""))
                    > CRITIC_ANSWER_CHAR_LIMIT
                ),
                "raw_answer": str(row.get("answer_text") or "")[
                    :CRITIC_ANSWER_CHAR_LIMIT
                ],
                "annotation": row.get("annotation"),
            }
            for row in rows
        ],
    }


def _scope_leakage_warning_machine_resolved(
    payload: dict[str, Any],
    warning: dict[str, Any],
) -> bool:
    """Recompute parent/portfolio hit vectors from canonical annotations.

    Equal aggregate rates are allowed only when both series can be rebuilt
    independently from complete raw rows: the parent vector needs a literal
    target alias, while the portfolio vector needs a separately grounded,
    profile-confirmed entity.  An LLM observation is never proof by itself.
    """

    profile = payload.get("site_profile")
    catalog = payload.get("entity_catalog")
    metrics = payload.get("candidate_metrics")
    answers = payload.get("answers")
    providers = warning.get("providers")
    if not (
        isinstance(profile, dict)
        and isinstance(catalog, dict)
        and isinstance(metrics, dict)
        and isinstance(answers, list)
        and isinstance(providers, list)
        and providers
    ):
        return False

    scoped_catalog = _scope_entity_catalog_to_profile(catalog, profile)
    target_aliases = _target_aliases(profile, scoped_catalog)
    confirmed_owner_aliases = _attribution_owner_aliases(
        profile,
        scoped_catalog,
    )
    entities: dict[str, dict[str, Any]] = {}
    for raw_entity in scoped_catalog.get("entities") or []:
        if not isinstance(raw_entity, dict) or not raw_entity.get(
            "canonical_name"
        ):
            continue
        entity = dict(raw_entity)
        if _catalog_marks_portfolio_entity(entity):
            entity["_profile_membership_confirmed"] = (
                _profile_confirms_portfolio_entity(entity, profile)
            )
            entity["_mention_policy"] = _portfolio_mention_policy(
                entity,
                profile,
            )
            entity["_attribution_aliases"] = _entity_attribution_aliases(
                profile,
                scoped_catalog,
                entity,
            )
            entity["_direct_target_aliases"] = target_aliases
            entity["_confirmed_owner_aliases"] = confirmed_owner_aliases
        entities[
            str(entity.get("canonical_name") or "").casefold()
        ] = entity

    provider_metrics = {
        str(item.get("name") or ""): item
        for item in metrics.get("providers") or []
        if isinstance(item, dict) and item.get("name")
    }
    provider_keys = {
        model.label: model.key for model in panel_models()
    }
    for provider_name in providers:
        if not isinstance(provider_name, str) or not provider_name.strip():
            return False
        provider_metric = provider_metrics.get(provider_name)
        provider_key = provider_keys.get(provider_name)
        if not isinstance(provider_metric, dict) or not provider_key:
            return False
        selected = [
            item
            for item in answers
            if isinstance(item, dict)
            and item.get("mode") == "web"
            and item.get("scenario_role") == "unbranded_discovery"
            and item.get("provider") == provider_key
        ]
        if not selected:
            return False
        if any(
            item.get("raw_answer_truncated") is True
            for item in selected
        ):
            return False
        completed_rows = [
            item
            for item in selected
            if item.get("status") == "completed"
            and item.get("metric_eligible") is not False
            and isinstance(item.get("raw_answer"), str)
            and str(item.get("raw_answer") or "").strip()
        ]
        valid_rows = [
            item
            for item in completed_rows
            if item.get("annotation_state") == "current"
            and item.get("raw_answer_truncated") is False
            and isinstance(item.get("annotation"), dict)
            and item["annotation"].get("valid") is True
        ]
        if len(valid_rows) != len(completed_rows):
            return False

        parent_ids: set[int] = set()
        portfolio_ids: set[int] = set()
        for item in valid_rows:
            answer_id = item.get("answer_id")
            if not isinstance(answer_id, int):
                return False
            raw_answer = str(item.get("raw_answer") or "")
            annotation = item["annotation"]
            if annotation.get("target_mentioned") is True:
                if not any(
                    _alias_is_present(raw_answer, alias)
                    for alias in target_aliases
                ):
                    return False
                parent_ids.add(answer_id)
            for mention in annotation.get("entity_mentions") or []:
                if not isinstance(mention, dict):
                    continue
                entity = entities.get(
                    str(mention.get("canonical_name") or "").casefold()
                ) or {}
                if (
                    _is_portfolio_target_entity(entity)
                    and _portfolio_entity_is_grounded(
                        raw_answer,
                        entity,
                        target_aliases,
                        mention,
                    )
                ):
                    portfolio_ids.add(answer_id)
                    break

        parent_metric = provider_metric.get("parent_discovery") or {}
        portfolio_metric = provider_metric.get("portfolio_capture") or {}
        if not all(
            isinstance(item, dict)
            for item in (parent_metric, portfolio_metric)
        ):
            return False
        expected_count = len(selected)
        completed_count = len(completed_rows)
        expected_state = (
            "complete"
            if expected_count and completed_count == expected_count
            else "limited"
            if completed_count
            else "unavailable"
        )
        for metric, hit_ids in (
            (parent_metric, parent_ids),
            (portfolio_metric, portfolio_ids),
        ):
            if (
                metric.get("data_state") != expected_state
                or metric.get("expected_answers") != expected_count
                or metric.get("completed_answers") != completed_count
                or metric.get("annotated_answers") != completed_count
                or metric.get("valid_answers") != completed_count
                or metric.get("coverage_rate")
                != _rate(completed_count, expected_count)
                or metric.get("mention_count") != len(hit_ids)
                or metric.get("mention_rate")
                != _rate(len(hit_ids), completed_count)
            ):
                return False
    return True


def _deterministic_warning_machine_resolved(
    payload: dict[str, Any],
    warning: dict[str, Any],
) -> bool:
    if warning.get("code") == "scope_leakage":
        return _scope_leakage_warning_machine_resolved(payload, warning)
    # Evidence mismatch and future material warnings stay fail-closed until a
    # code-specific deterministic resolver exists for their exact contract.
    return False


def _critic_review_errors(
    review: dict[str, Any],
    *,
    payload: dict[str, Any] | None = None,
) -> list[str]:
    """Revalidate cached and live reviews before they can open the gate."""

    errors: list[str] = []
    required_fields = {
        "verdict",
        "summary",
        "anomalies",
        "policy_adjustments",
        "annotation_guidance",
        "acceptance_checks",
    }
    for field in sorted(required_fields):
        if field not in review:
            errors.append(f"missing required critic field: {field}")
        elif review.get(field) is None:
            errors.append(f"null required critic field: {field}")
    verdict = review.get("verdict")
    if verdict not in {"pass", "revise", "block"}:
        errors.append("invalid verdict")
    summary = review.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        errors.append("summary must be a non-empty string")
    anomalies = review.get("anomalies")
    adjustments = review.get("policy_adjustments")
    if not isinstance(anomalies, list):
        errors.append("anomalies must be a list")
        anomalies = []
    if not isinstance(adjustments, list):
        errors.append("policy_adjustments must be a list")
        adjustments = []
    guidance = review.get("annotation_guidance")
    if not isinstance(guidance, str):
        errors.append("annotation_guidance must be a string")
        guidance = ""
    acceptance_checks = review.get("acceptance_checks")
    if not isinstance(acceptance_checks, list):
        errors.append("acceptance_checks must be a list")
        acceptance_checks = []
    elif any(
        not isinstance(check, str) or not check.strip()
        for check in acceptance_checks
    ):
        errors.append("acceptance_checks must contain non-empty strings")

    allowed_anomaly_codes = {
        "scope_leakage",
        "generic_term_leakage",
        "unsupported_membership",
        "fabricated_evidence",
        "annotation_evidence_mismatch",
        "brand_knowledge_false_negative",
        "provider_uniformity",
        "denominator_error",
        "missing_data_as_zero",
        "other",
    }
    for index, anomaly in enumerate(anomalies):
        if not isinstance(anomaly, dict):
            errors.append(f"anomaly {index} must be an object")
            continue
        if anomaly.get("code") not in allowed_anomaly_codes:
            errors.append(f"anomaly {index} has invalid code")
        if anomaly.get("severity") not in {
            "critical",
            "important",
            "observation",
        }:
            errors.append(f"anomaly {index} has invalid severity")
        if not isinstance(anomaly.get("finding"), str) or not str(
            anomaly.get("finding") or ""
        ).strip():
            errors.append(f"anomaly {index} has invalid finding")
        if not isinstance(anomaly.get("answer_ids"), list) or any(
            not isinstance(value, int)
            for value in anomaly.get("answer_ids") or []
        ):
            errors.append(f"anomaly {index} has invalid answer_ids")
        if not isinstance(anomaly.get("entities"), list) or any(
            not isinstance(value, str)
            for value in anomaly.get("entities") or []
        ):
            errors.append(f"anomaly {index} has invalid entities")
    truncated_answer_ids = [
        item.get("answer_id")
        for item in (payload or {}).get("answers") or []
        if isinstance(item, dict)
        and item.get("raw_answer_truncated") is True
    ]
    if truncated_answer_ids and verdict != "block":
        errors.append(
            "truncated raw answers require block: "
            + ", ".join(str(value) for value in truncated_answer_ids)
        )
    if verdict == "pass":
        unresolved = [
            item
            for item in anomalies
            if isinstance(item, dict)
            and item.get("severity") in {"critical", "important"}
        ]
        if unresolved:
            errors.append("pass contains unresolved critical/important anomalies")
        if adjustments:
            errors.append("pass contains pending policy adjustments")
        if str(review.get("annotation_guidance") or "").strip():
            errors.append("pass contains pending annotation guidance")
        if not acceptance_checks:
            errors.append("pass contains no acceptance checks")
        material_warnings = [
            warning
            for warning in (payload or {}).get("deterministic_warnings") or []
            if isinstance(warning, dict)
            and warning.get("severity") in {"critical", "important"}
        ]
        warning_codes = [
            str(warning.get("code") or "") for warning in material_warnings
        ]
        if len(warning_codes) != len(set(warning_codes)):
            errors.append(
                "pass cannot resolve duplicate material warning codes "
                "without unique machine references"
            )
        for warning in material_warnings:
            warning_code = str(warning.get("code") or "")
            matching_observations = [
                item
                for item in anomalies
                if isinstance(item, dict)
                and item.get("severity") == "observation"
                and str(item.get("code") or "") == warning_code
            ]
            if len(matching_observations) != 1:
                errors.append(
                    "pass did not explain deterministic warning: "
                    + (warning_code or "unknown")
                )
                continue
            observation = matching_observations[0]
            finding = str(observation.get("finding") or "").strip()
            if len(finding) < 40 or len(finding.split()) < 6:
                errors.append(
                    "deterministic warning explanation is not substantive: "
                    + (warning_code or "unknown")
                )
            warning_answer_ids = {
                value
                for value in warning.get("answer_ids") or []
                if isinstance(value, int)
            }
            observation_answer_ids = {
                value
                for value in observation.get("answer_ids") or []
                if isinstance(value, int)
            }
            if not warning_answer_ids.issubset(observation_answer_ids):
                errors.append(
                    "deterministic warning explanation misses answer ids: "
                    + (warning_code or "unknown")
                )
            warning_entities = {
                str(value).casefold()
                for value in warning.get("entities") or []
                if isinstance(value, str) and value.strip()
            }
            observation_entities = {
                str(value).casefold()
                for value in observation.get("entities") or []
                if isinstance(value, str) and value.strip()
            }
            if not warning_entities.issubset(observation_entities):
                errors.append(
                    "deterministic warning explanation misses entities: "
                    + (warning_code or "unknown")
                )
            if not _deterministic_warning_machine_resolved(
                payload or {},
                warning,
            ):
                errors.append(
                    "deterministic warning lacks machine resolution: "
                    + (warning_code or "unknown")
                )
    elif verdict == "revise":
        material_anomalies = [
            item
            for item in anomalies
            if isinstance(item, dict)
            and item.get("severity") in {"critical", "important"}
        ]
        if not material_anomalies:
            errors.append("revise contains no critical/important anomalies")
        if not adjustments:
            errors.append("revise contains no policy adjustments")
        if not guidance.strip():
            errors.append("revise contains no annotation guidance")
    return errors


def _critic_review_validation_errors(
    review: dict[str, Any],
    *,
    payload: dict[str, Any],
) -> list[str]:
    """Validate both the decision contract and safe actionability."""

    errors = _critic_review_errors(review, payload=payload)
    if review.get("verdict") != "revise":
        return errors
    adjustments = review.get("policy_adjustments")
    if not isinstance(adjustments, list) or not adjustments:
        return errors

    valid_answer_ids = {
        int(answer["answer_id"])
        for answer in payload.get("answers") or []
        if isinstance(answer, dict)
        and isinstance(answer.get("answer_id"), int)
        and answer.get("status") == "completed"
        and answer.get("metric_eligible", True) is not False
        and str(answer.get("raw_answer") or "").strip()
        and isinstance(answer.get("annotation"), dict)
        and answer["annotation"].get("valid") is True
    }
    _catalog, applied, _guidance = _apply_critic_policy(
        payload.get("entity_catalog") or {},
        review,
        valid_answer_ids=valid_answer_ids,
    )
    if not applied:
        errors.append(
            "revise contains no safely applicable policy adjustments"
        )
    return errors


def _deterministic_critic_fallback_review(
    payload: dict[str, Any],
    incomplete_review: dict[str, Any],
    *,
    validation_errors: list[str],
) -> dict[str, Any]:
    """Fail open only when code independently proves a model ``pass`` safe."""

    anomalies = incomplete_review.get("anomalies")
    adjustments = incomplete_review.get("policy_adjustments")
    guidance = incomplete_review.get("annotation_guidance")
    material_anomalies = [
        item
        for item in anomalies or []
        if isinstance(item, dict)
        and item.get("severity") in {"critical", "important"}
    ]
    observations_are_bounded = bool(
        isinstance(anomalies, list)
        and all(
            isinstance(item, dict)
            and item.get("severity") == "observation"
            for item in anomalies
        )
    )
    material_warnings = [
        warning
        for warning in payload.get("deterministic_warnings") or []
        if isinstance(warning, dict)
        and warning.get("severity") in {"critical", "important"}
    ]
    warning_codes = [
        str(warning.get("code") or "") for warning in material_warnings
    ]
    warnings_are_resolved = bool(
        len(warning_codes) == len(set(warning_codes))
        and all(
            _deterministic_warning_machine_resolved(payload, warning)
            for warning in material_warnings
        )
    )
    has_truncated_raw = any(
        isinstance(item, dict)
        and item.get("raw_answer_truncated") is True
        for item in payload.get("answers") or []
    )
    safe_pass = bool(
        incomplete_review.get("verdict") == "pass"
        and observations_are_bounded
        and not material_anomalies
        and isinstance(adjustments, list)
        and not adjustments
        and isinstance(guidance, str)
        and not guidance.strip()
        and not has_truncated_raw
        and warnings_are_resolved
    )
    if safe_pass:
        fallback = {
            "verdict": "pass",
            "summary": (
                "Формат ответа независимого критика восстановить не удалось, "
                "но кодовая проверка повторно собрала метрики из raw-ответов "
                "и подтвердила все материальные предупреждения."
            ),
            "anomalies": [
                {
                    "code": str(warning.get("code") or "other"),
                    "severity": "observation",
                    "finding": (
                        "Кодовая проверка заново собрала затронутые векторы "
                        "по raw-ответам и подтвердила отсутствие искажения "
                        "публикуемой метрики."
                    ),
                    "answer_ids": [
                        value
                        for value in warning.get("answer_ids") or []
                        if isinstance(value, int)
                    ],
                    "entities": [
                        value
                        for value in warning.get("entities") or []
                        if isinstance(value, str) and value.strip()
                    ],
                }
                for warning in material_warnings
            ],
            "policy_adjustments": [],
            "annotation_guidance": "",
            "acceptance_checks": [
                "Кодом повторно сверены знаменатели, числители и состояния "
                "полноты по исходным ответам."
            ],
            "fallback": {
                "kind": "deterministic_safe_pass",
                "critic_validation_errors": validation_errors[:20],
            },
        }
    else:
        fallback = {
            "verdict": "block",
            "summary": (
                "Формат ответа независимого критика не восстановлен, а "
                "кодовая проверка не смогла безопасно подтвердить все "
                "материальные условия публикации."
            ),
            "anomalies": [],
            "policy_adjustments": [],
            "annotation_guidance": "",
            "acceptance_checks": [],
            "fallback": {
                "kind": "deterministic_block",
                "critic_validation_errors": validation_errors[:20],
            },
        }
    fallback_errors = _critic_review_validation_errors(
        fallback,
        payload=payload,
    )
    if fallback_errors:
        raise AssertionError(
            "Deterministic critic fallback violated its own contract: "
            + "; ".join(fallback_errors)
        )
    return fallback


async def _analysis_critic_fallback_artifact(
    run_id: str,
    *,
    iteration: int,
    payload: dict[str, Any],
    incomplete_review: dict[str, Any],
    validation_errors: list[str],
) -> dict[str, Any]:
    """Persist the bounded non-LLM fallback after one failed repair."""

    artifact_key = f"analysis_critic_r{iteration}_fallback"
    fallback_input = {
        "incomplete_review": incomplete_review,
        "validation_errors": validation_errors,
        "deterministic_warnings": payload.get("deterministic_warnings") or [],
    }
    fallback = _deterministic_critic_fallback_review(
        payload,
        incomplete_review,
        validation_errors=validation_errors,
    )
    await _save_artifact(
        run_id,
        stage_key="knowledge_gap",
        artifact_key=artifact_key,
        status="completed",
        model="deterministic/critic-fallback-v1",
        input_json=fallback_input,
        output_json=fallback,
        error_message=None,
        prompt_version=ANALYSIS_CRITIC_VERSION,
    )
    return fallback


async def _analysis_critic_repair_artifact(
    run_id: str,
    *,
    iteration: int,
    payload: dict[str, Any],
    incomplete_review: dict[str, Any],
    validation_errors: list[str],
) -> dict[str, Any]:
    """Persist one bounded schema/semantics repair of a critic decision."""

    artifact_key = f"analysis_critic_r{iteration}_repair"
    repair_input = {
        "audit_payload": payload,
        "incomplete_review": incomplete_review,
        "validation_errors": validation_errors,
    }
    cached = await _artifact_output(
        run_id,
        artifact_key,
        input_json=repair_input,
        model=CRITIC_MODEL,
        prompt_version=ANALYSIS_CRITIC_VERSION,
    )
    if isinstance(cached, dict):
        errors = _critic_review_validation_errors(
            cached,
            payload=payload,
        )
        if errors:
            return await _analysis_critic_fallback_artifact(
                run_id,
                iteration=iteration,
                payload=payload,
                incomplete_review=incomplete_review,
                validation_errors=[
                    "Cached analysis critic repair is inconsistent: "
                    + "; ".join(errors)
                ],
            )
        return cached

    await _save_artifact(
        run_id,
        stage_key="knowledge_gap",
        artifact_key=artifact_key,
        status="running",
        model=CRITIC_MODEL,
        input_json=repair_input,
        prompt_version=ANALYSIS_CRITIC_VERSION,
    )
    repaired: dict[str, Any] | None = None
    raw_text: str | None = None
    usage: dict[str, Any] | None = None
    try:
        repaired, raw_text, usage = await repair_analysis_review(
            payload,
            incomplete_review,
            iteration=iteration,
            validation_errors=validation_errors,
        )
        errors = _critic_review_validation_errors(
            repaired,
            payload=payload,
        )
        if errors:
            raise OpenRouterError(
                "Analysis critic repair is inconsistent: "
                + "; ".join(errors)
            )
    except Exception as exc:
        await _save_artifact(
            run_id,
            stage_key="knowledge_gap",
            artifact_key=artifact_key,
            status="failed",
            model=CRITIC_MODEL,
            input_json=repair_input,
            output_json=repaired,
            raw_text=raw_text,
            usage_json=usage,
            error_message=str(exc),
            prompt_version=ANALYSIS_CRITIC_VERSION,
        )
        return await _analysis_critic_fallback_artifact(
            run_id,
            iteration=iteration,
            payload=payload,
            incomplete_review=incomplete_review,
            validation_errors=[*validation_errors, str(exc)],
        )
    await _save_artifact(
        run_id,
        stage_key="knowledge_gap",
        artifact_key=artifact_key,
        status="completed",
        model=CRITIC_MODEL,
        input_json=repair_input,
        output_json=repaired,
        raw_text=raw_text,
        usage_json=usage,
        prompt_version=ANALYSIS_CRITIC_VERSION,
    )
    return repaired


async def _analysis_critic_artifact(
    run_id: str,
    *,
    iteration: int,
    payload: dict[str, Any],
) -> dict[str, Any]:
    artifact_key = f"analysis_critic_r{iteration}"
    cached = await _artifact_output(
        run_id,
        artifact_key,
        input_json=payload,
        model=CRITIC_MODEL,
        prompt_version=ANALYSIS_CRITIC_VERSION,
    )
    if isinstance(cached, dict):
        errors = _critic_review_validation_errors(
            cached,
            payload=payload,
        )
        if errors:
            return await _analysis_critic_repair_artifact(
                run_id,
                iteration=iteration,
                payload=payload,
                incomplete_review=cached,
                validation_errors=errors,
            )
        return cached

    await _save_artifact(
        run_id,
        stage_key="knowledge_gap",
        artifact_key=artifact_key,
        status="running",
        model=CRITIC_MODEL,
        input_json=payload,
        prompt_version=ANALYSIS_CRITIC_VERSION,
    )
    review: dict[str, Any] | None = None
    raw_text: str | None = None
    usage: dict[str, Any] | None = None
    try:
        review, raw_text, usage = await review_analysis(
            payload,
            iteration=iteration,
        )
        errors = _critic_review_validation_errors(
            review,
            payload=payload,
        )
        if errors:
            review = await _analysis_critic_repair_artifact(
                run_id,
                iteration=iteration,
                payload=payload,
                incomplete_review=review,
                validation_errors=errors,
            )
    except Exception as exc:
        await _save_artifact(
            run_id,
            stage_key="knowledge_gap",
            artifact_key=artifact_key,
            status="failed",
            model=CRITIC_MODEL,
            input_json=payload,
            output_json=review,
            raw_text=raw_text,
            usage_json=usage,
            error_message=str(exc),
            prompt_version=ANALYSIS_CRITIC_VERSION,
        )
        raise
    await _save_artifact(
        run_id,
        stage_key="knowledge_gap",
        artifact_key=artifact_key,
        status="completed",
        model=CRITIC_MODEL,
        input_json=payload,
        output_json=review,
        raw_text=raw_text,
        usage_json=usage,
        prompt_version=ANALYSIS_CRITIC_VERSION,
    )
    return review


def _require_entity_attribution(entity: dict[str, Any]) -> None:
    entity["mention_policy"] = "requires_target_attribution"
    rewritten: list[Any] = []
    for raw_alias in entity.get("aliases") or []:
        alias = _catalog_alias_value(raw_alias)
        if not alias:
            continue
        rewritten.append(
            {
                "value": alias,
                "match_policy": "requires_target_attribution",
            }
        )
    entity["aliases"] = rewritten


def _require_alias_attribution(
    entity: dict[str, Any],
    alias_to_tighten: str,
) -> bool:
    normalized_target = (
        alias_to_tighten.casefold().replace("ё", "е").strip()
    )
    if not normalized_target:
        return False
    canonical = str(entity.get("canonical_name") or "").strip()
    if canonical.casefold().replace("ё", "е") == normalized_target:
        entity["mention_policy"] = "requires_target_attribution"
        return True

    found = False
    rewritten: list[Any] = []
    for raw_alias in entity.get("aliases") or []:
        alias = _catalog_alias_value(raw_alias)
        if alias.casefold().replace("ё", "е") == normalized_target:
            rewritten.append(
                {
                    "value": alias,
                    "match_policy": "requires_target_attribution",
                }
            )
            found = True
        else:
            rewritten.append(raw_alias)
    entity["aliases"] = rewritten
    return found


def _apply_critic_policy(
    catalog: dict[str, Any],
    review: dict[str, Any],
    *,
    valid_answer_ids: set[int],
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    """Apply only narrowing, enumerated per-run changes proposed by the critic."""

    tightened = copy.deepcopy(catalog)
    tightened["entities"] = _deduplicate_scoped_catalog_entities(
        list(tightened.get("entities") or [])
    )
    entities = {
        str(entity.get("canonical_name") or "").casefold(): entity
        for entity in tightened.get("entities") or []
        if isinstance(entity, dict) and entity.get("canonical_name")
    }
    applied: list[dict[str, Any]] = []
    for raw_adjustment in review.get("policy_adjustments") or []:
        if not isinstance(raw_adjustment, dict):
            continue
        answer_ids = {
            int(value)
            for value in raw_adjustment.get("answer_ids") or []
            if isinstance(value, int) and value in valid_answer_ids
        }
        if not answer_ids:
            continue
        name = str(raw_adjustment.get("entity_name") or "").strip()
        entity = entities.get(name.casefold())
        if entity is None:
            continue
        action = str(raw_adjustment.get("action") or "")
        changed = False
        if action == "exclude_portfolio_entity":
            entity["commercially_relevant"] = False
            entity["_critic_excluded"] = True
            changed = True
        elif action == "require_target_attribution":
            _require_entity_attribution(entity)
            changed = True
        elif action == "require_alias_attribution":
            changed = _require_alias_attribution(
                entity,
                str(raw_adjustment.get("alias") or ""),
            )
        elif action == "require_literal_attribution_evidence":
            changed = _catalog_marks_portfolio_entity(entity)
        elif action == "require_literal_brand_knowledge_evidence":
            relationship = str(
                entity.get("target_relationship")
                or entity.get("relationship")
                or ""
            ).casefold()
            changed = (
                entity.get("category") == "target"
                and relationship in {"exact_target", "self"}
            )
        if changed:
            applied.append(
                {
                    "action": action,
                    "entity_name": name,
                    "alias": raw_adjustment.get("alias"),
                    "reason": str(raw_adjustment.get("reason") or "")[:1000],
                    "answer_ids": sorted(answer_ids),
                }
            )

    guidance_lines = [
        "Дополнительные обязательные ограничения независимого критика:",
        *[
            (
                f"- {item['entity_name']}: "
                + (
                    "не учитывать как часть целевого портфеля"
                    if item["action"] == "exclude_portfolio_entity"
                    else (
                        (
                            "для ответов "
                            + ", ".join(
                                str(value)
                                for value in item["answer_ids"]
                            )
                            + " заново проверить brand_answer по буквальным "
                            "фактам о целевом бренде; specific допустим "
                            "только для конкретных фактов, согласующихся "
                            "с site_profile или entity_catalog"
                        )
                        if item["action"]
                        == "require_literal_brand_knowledge_evidence"
                        else
                        (
                            "при явной связи в raw-ответе скопировать один "
                            "точный непрерывный фрагмент с разрешённым "
                            "владельцем, услугой и словами связи, сохранив "
                            "Markdown и регистр; иначе не атрибутировать"
                        )
                        if item["action"]
                        == "require_literal_attribution_evidence"
                        else (
                            "требовать явную буквальную атрибуцию для алиаса "
                            f"«{item['alias']}»"
                            if item["action"] == "require_alias_attribution"
                            else (
                                "требовать явную буквальную атрибуцию "
                                "целевому бренду"
                            )
                        )
                    )
                )
                + "."
            )
            for item in applied
        ],
    ]
    guidance = "\n".join(guidance_lines) if applied else ""
    return tightened, applied, guidance


async def _save_critic_gate(
    run_id: str,
    *,
    passed: bool,
    iteration: int,
    profile: dict[str, Any],
    catalog: dict[str, Any],
    rows: list[dict[str, Any]],
    metrics: dict[str, Any],
    policy_history: list[dict[str, Any]],
    reason: str,
    expected_corpus_cells: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    provenance = _critic_provenance_digests(
        profile=profile,
        catalog=catalog,
        rows=rows,
        metrics=metrics,
        policy_history=policy_history,
    )
    corpus_manifest = _final_corpus_manifest(
        rows,
        expected_cells=(
            expected_corpus_cells
            if expected_corpus_cells is not None
            else _expected_corpus_cells_from_rows(rows)
        ),
    )
    output = {
        "passed": passed,
        "iteration": iteration,
        "critic_model": CRITIC_MODEL,
        "metrics_sha256": provenance["metrics_sha256"],
        "provenance": provenance,
        "corpus_manifest": corpus_manifest,
        "policy_history": policy_history,
        "reason": reason[:2000],
    }
    await _save_artifact(
        run_id,
        stage_key="knowledge_gap",
        artifact_key="analysis_critic_gate",
        status="completed" if passed else "failed",
        model=CRITIC_MODEL,
        input_json={
            "provenance": provenance,
            "corpus_manifest_digest": corpus_manifest["digest"],
            "iteration": iteration,
        },
        output_json=output,
        error_message=None if passed else output["reason"],
        prompt_version=ANALYSIS_CRITIC_VERSION,
    )
    return output


async def _run_analysis_critic_loop(
    run_id: str,
    *,
    profile: dict[str, Any],
    catalog: dict[str, Any],
    rows: list[dict[str, Any]],
    metrics: dict[str, Any],
    expected_corpus_cells: list[dict[str, Any]] | None = None,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    """Gate metrics with at most one repair and two independent reviews."""

    current_catalog = _scope_entity_catalog_to_profile(catalog, profile)
    current_rows = rows
    current_metrics = metrics
    policy_history: list[dict[str, Any]] = []
    accumulated_guidance: list[str] = []
    valid_answer_ids = {
        int(row["answer_id"])
        for row in rows
        if isinstance(row.get("answer_id"), int)
        and _row_has_completed_raw(row)
        and isinstance(row.get("annotation"), dict)
        and row["annotation"].get("valid") is True
    }

    for iteration in range(1, MAX_CRITIC_ITERATIONS + 1):
        await update_progress(
            run_id,
            stage="knowledge_gap",
            percent=80,
            detail=(
                "Независимый критик сверяет метрики с исходными ответами "
                f"({iteration}/{MAX_CRITIC_ITERATIONS})."
            ),
            eta_seconds=240,
        )
        payload = _critic_payload(
            profile=profile,
            catalog=current_catalog,
            rows=current_rows,
            metrics=current_metrics,
            policy_history=policy_history,
        )
        review = await _analysis_critic_artifact(
            run_id,
            iteration=iteration,
            payload=payload,
        )
        verdict = str(review.get("verdict") or "block")
        if verdict == "pass":
            gate = await _save_critic_gate(
                run_id,
                passed=True,
                iteration=iteration,
                profile=profile,
                catalog=current_catalog,
                rows=current_rows,
                metrics=current_metrics,
                policy_history=policy_history,
                reason=str(review.get("summary") or "Проверка пройдена."),
                expected_corpus_cells=expected_corpus_cells,
            )
            return current_catalog, current_rows, current_metrics, gate

        if verdict == "block" or iteration >= MAX_CRITIC_ITERATIONS:
            reason = str(
                review.get("summary")
                or "Критик не подтвердил корректность аналитики."
            )
            await _save_critic_gate(
                run_id,
                passed=False,
                iteration=iteration,
                profile=profile,
                catalog=current_catalog,
                rows=current_rows,
                metrics=current_metrics,
                policy_history=policy_history,
                reason=reason,
                expected_corpus_cells=expected_corpus_cells,
            )
            raise OpenRouterError(
                "Analysis critic blocked report publication: " + reason
            )

        tightened, applied, guidance = _apply_critic_policy(
            current_catalog,
            review,
            valid_answer_ids=valid_answer_ids,
        )
        if guidance:
            accumulated_guidance.append(guidance)
        if not applied and not accumulated_guidance:
            reason = (
                "Критик запросил переработку, но не предложил безопасного "
                "ужесточения политики."
            )
            await _save_critic_gate(
                run_id,
                passed=False,
                iteration=iteration,
                profile=profile,
                catalog=current_catalog,
                rows=current_rows,
                metrics=current_metrics,
                policy_history=policy_history,
                reason=reason,
                expected_corpus_cells=expected_corpus_cells,
            )
            raise OpenRouterError(reason)

        policy_step = {
            "iteration": iteration,
            "summary": str(review.get("summary") or "")[:2000],
            "adjustments": applied,
            "annotation_guidance": guidance,
        }
        policy_history.append(policy_step)
        current_catalog = tightened
        await _save_artifact(
            run_id,
            stage_key="knowledge_gap",
            artifact_key="analysis_critic_policy",
            status="completed",
            model=CRITIC_MODEL,
            input_json={"iteration": iteration, "review": review},
            output_json={
                "base_catalog_version": ENTITY_CATALOG_VERSION,
                "policy_history": policy_history,
                "effective_catalog": current_catalog,
            },
            prompt_version=ANALYSIS_CRITIC_VERSION,
        )
        await _annotate_answers(
            run_id,
            profile,
            current_catalog,
            research_guidance="\n\n".join(accumulated_guidance),
        )
        current_rows = await _metric_rows(
            run_id,
            annotation_input_sha256=_annotation_context_sha256(
                profile,
                current_catalog,
                "\n\n".join(accumulated_guidance),
            ),
        )
        current_metrics = _compute_metrics(
            current_rows,
            profile,
            current_catalog,
        )

    raise AssertionError("Unreachable critic-loop state")


async def _evidence_sample(run_id: str) -> list[dict[str, Any]]:
    legacy_contract = await _legacy_panel_run_contract(run_id)
    async with SessionLocal() as session:
        rows = (
            await session.execute(
                select(ModelAnswer, VisibilityPrompt, AnswerAnnotation)
                .join(VisibilityPrompt, ModelAnswer.prompt_id == VisibilityPrompt.id)
                .join(AnswerAnnotation, AnswerAnnotation.answer_id == ModelAnswer.id)
                .where(ModelAnswer.run_id == run_id)
                .order_by(ModelAnswer.id)
            )
        ).all()
    labels = {model.key: model.label for model in panel_models()}
    evidence: list[dict[str, Any]] = []
    for answer, prompt, annotation in rows:
        answer_text = str(answer.response_text or "")
        attested, _reason = _panel_answer_attestation(
            answer,
            prompt_text=prompt.text,
            legacy_allowed=legacy_contract["eligible"],
        )
        if answer.status != "completed" or not answer_text.strip() or not attested:
            continue
        evidence.append(
            {
                "mode": answer.mode,
                "system": labels.get(answer.provider_key, answer.provider_key),
                "intent": prompt.intent_class,
                "role": prompt.role,
                "target_role": annotation.annotation_json.get("target_role"),
                "sentiment": annotation.annotation_json.get("sentiment"),
                "evidence": (annotation.annotation_json.get("evidence") or [])[:2],
                "uncertainties": (
                    annotation.annotation_json.get("uncertainties") or []
                )[:2],
            }
        )
        if len(evidence) >= 60:
            break
    return evidence


def _full_answer_corpus_item(
    answer: ModelAnswer,
    prompt: VisibilityPrompt,
    annotation: AnswerAnnotation | None,
    *,
    legacy_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Preserve one complete scenario/answer pair and all its provenance."""

    labels = {model.key: model.label for model in panel_models()}
    annotation_json = (
        annotation.annotation_json
        if annotation is not None
        and isinstance(annotation.annotation_json, dict)
        else {}
    )
    answer_text = str(answer.response_text or "")
    panel_contract = (answer.usage_json or {}).get("_aiv_panel_contract")
    legacy_context = legacy_contract or {"eligible": False, "digest": ""}
    web_attested, web_attestation_reason = _panel_answer_attestation(
        answer,
        prompt_text=prompt.text,
        legacy_allowed=legacy_context["eligible"],
    )
    metric_access = _panel_metric_access(
        answer,
        transport_attested=web_attested,
        attestation_reason=web_attestation_reason,
        legacy_memory_observation_allowed=legacy_context.get(
            "memory_observation_eligible",
            False,
        ),
    )
    answer_complete = bool(
        answer.status == "completed" and answer_text.strip()
    )
    return {
        "answer_id": answer.id,
        "prompt_id": prompt.id,
        "prompt_key": prompt.prompt_key,
        "scenario": prompt.text,
        "scenario_sequence": prompt.sequence,
        "scenario_role": prompt.role,
        "intent_class": prompt.intent_class,
        "scenario_rationale": prompt.rationale,
        "provider_key": answer.provider_key,
        "system": labels.get(answer.provider_key, answer.provider_key),
        "model": answer.model,
        "mode": answer.mode,
        "status": answer.status,
        "citations": answer.citations_json or [],
        "metric_eligible": bool(
            answer_complete and metric_access["metric_eligible"]
        ),
        "context_eligible": bool(
            answer_complete and metric_access["context_eligible"]
        ),
        "metric_evidence_state": metric_access["metric_evidence_state"],
        "metric_limitation": metric_access["metric_limitation"],
        "panel_evidence": {
            "version": (
                LEGACY_PANEL_EVIDENCE_VERSION
                if web_attestation_reason.startswith("legacy_")
                else WEB_ATTESTATION_VERSION
            ),
            "reason": web_attestation_reason,
            "sha256": _panel_evidence_sha256(
                answer,
                reason=web_attestation_reason,
                legacy_contract=legacy_context,
            ),
        },
        "annotation": annotation_json,
        "answer_text": answer_text,
        "provenance": {
            "raw_answer_sha256": hashlib.sha256(
                answer_text.encode("utf-8")
            ).hexdigest(),
            "annotation_sha256": _stable_json_sha256(annotation_json),
            "annotation_version": annotation_json.get("_annotation_version"),
            "annotation_answer_sha256": annotation_json.get("_answer_sha256"),
            "annotation_answer_model": annotation_json.get("_answer_model"),
            "annotation_input_sha256": annotation_json.get(
                "_annotation_input_sha256"
            ),
            "panel_contract": (
                panel_contract if isinstance(panel_contract, dict) else None
            ),
        },
    }


def _rows_from_full_answer_models(
    rows: list[
        tuple[ModelAnswer, VisibilityPrompt, AnswerAnnotation | None]
    ],
    *,
    legacy_contract: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for answer, prompt, annotation in rows:
        annotation_json = (
            annotation.annotation_json
            if annotation is not None
            and isinstance(annotation.annotation_json, dict)
            else {}
        )
        answer_text = str(answer.response_text or "")
        annotation_input_sha256 = str(
            annotation_json.get("_annotation_input_sha256") or ""
        )
        annotation_is_current = bool(
            answer.status == "completed"
            and answer_text.strip()
            and annotation_input_sha256
            and _annotation_matches_answer(
                annotation_json,
                answer_text=answer_text,
                answer_model=answer.model,
                annotation_input_sha256=annotation_input_sha256,
            )
        )
        legacy_context = legacy_contract or {
            "eligible": False,
            "digest": "",
        }
        web_attested, web_attestation_reason = _panel_answer_attestation(
            answer,
            prompt_text=prompt.text,
            legacy_allowed=legacy_context["eligible"],
        )
        metric_access = _panel_metric_access(
            answer,
            transport_attested=web_attested,
            attestation_reason=web_attestation_reason,
            legacy_memory_observation_allowed=legacy_context.get(
                "memory_observation_eligible",
                False,
            ),
        )
        answer_complete = bool(
            answer.status == "completed" and answer_text.strip()
        )
        output.append(
            {
                "answer_id": answer.id,
                "mode": answer.mode,
                "provider_key": answer.provider_key,
                "prompt_id": prompt.id,
                "prompt_key": prompt.prompt_key,
                "scenario": prompt.text,
                "intent_class": prompt.intent_class,
                "role": prompt.role,
                "status": answer.status,
                "model": answer.model,
                "citations_count": len(answer.citations_json or []),
                "answer_text": answer_text,
                "web_attested": web_attested,
                "web_attestation_reason": web_attestation_reason,
                "panel_evidence_version": (
                    LEGACY_PANEL_EVIDENCE_VERSION
                    if web_attestation_reason.startswith("legacy_")
                    else WEB_ATTESTATION_VERSION
                ),
                "panel_evidence_sha256": _panel_evidence_sha256(
                    answer,
                    reason=web_attestation_reason,
                    legacy_contract=legacy_context,
                ),
                "metric_eligible": bool(
                    answer_complete and metric_access["metric_eligible"]
                ),
                "context_eligible": bool(
                    answer_complete and metric_access["context_eligible"]
                ),
                "metric_evidence_state": metric_access[
                    "metric_evidence_state"
                ],
                "metric_limitation": metric_access["metric_limitation"],
                "annotation": (
                    annotation_json if annotation_is_current else {}
                ),
                "annotation_state": (
                    "current"
                    if annotation_is_current
                    else "ineligible"
                    if answer.status != "completed" or not answer_text.strip()
                    else "missing_or_stale"
                ),
            }
        )
    return output


def _full_answer_corpus_items(
    rows: list[
        tuple[ModelAnswer, VisibilityPrompt, AnswerAnnotation | None]
    ],
    *,
    legacy_contract: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    items = [
        _full_answer_corpus_item(
            answer,
            prompt,
            annotation,
            legacy_contract=legacy_contract,
        )
        for answer, prompt, annotation in rows
    ]
    return sorted(
        items,
        key=lambda item: (
            int(item.get("prompt_id") or 0),
            str(item.get("provider_key") or ""),
            str(item.get("model") or ""),
            str(item.get("mode") or ""),
            int(item.get("answer_id") or 0),
        ),
    )


def _selection_dimension_values(item: dict[str, Any]) -> dict[str, str]:
    annotation = (
        item.get("annotation")
        if isinstance(item.get("annotation"), dict)
        else {}
    )
    panel_evidence = (
        item.get("panel_evidence")
        if isinstance(item.get("panel_evidence"), dict)
        else {}
    )
    context_eligible = item.get(
        "context_eligible",
        item.get("metric_eligible", True),
    ) is not False
    dimensions = {
        "provider_mode": (
            f"{str(item.get('provider_key') or 'unknown')}|"
            f"{str(item.get('mode') or 'unknown')}"
        ),
        "intent_class": str(item.get("intent_class") or "unknown"),
        "scenario_role": str(item.get("scenario_role") or "unknown"),
        "evidence_state": str(panel_evidence.get("reason") or "unknown"),
        "context_access": "full_text" if context_eligible else "metadata_only",
    }
    if not context_eligible:
        return dimensions
    valid = annotation.get("valid")
    target_mentioned = annotation.get("target_mentioned")
    return {
        **dimensions,
        "target_role": str(annotation.get("target_role") or "unknown"),
        "sentiment": str(annotation.get("sentiment") or "unknown"),
        "valid": (
            "true" if valid is True else "false" if valid is False else "unknown"
        ),
        "target_mentioned": (
            "true"
            if target_mentioned is True
            else "false"
            if target_mentioned is False
            else "unknown"
        ),
    }


def _selection_coverage(items: list[dict[str, Any]]) -> dict[str, list[str]]:
    coverage: dict[str, set[str]] = defaultdict(set)
    for item in items:
        for dimension, value in _selection_dimension_values(item).items():
            coverage[dimension].add(value)
    return {
        dimension: sorted(values)
        for dimension, values in sorted(coverage.items())
    }


def _selection_features(item: dict[str, Any]) -> set[tuple[str, str]]:
    return set(_selection_dimension_values(item).items())


def _selection_item_signature(item: dict[str, Any]) -> dict[str, Any]:
    annotation = (
        item.get("annotation")
        if isinstance(item.get("annotation"), dict)
        else {}
    )
    provenance = (
        item.get("provenance")
        if isinstance(item.get("provenance"), dict)
        else {}
    )
    panel_evidence = (
        item.get("panel_evidence")
        if isinstance(item.get("panel_evidence"), dict)
        else {}
    )
    answer_text = str(item.get("answer_text") or "")
    return {
        "answer_id": item.get("answer_id"),
        "prompt_id": item.get("prompt_id"),
        "provider_key": item.get("provider_key"),
        "mode": item.get("mode"),
        "raw_answer_sha256": (
            provenance.get("raw_answer_sha256")
            or hashlib.sha256(answer_text.encode("utf-8")).hexdigest()
        ),
        "annotation_sha256": (
            provenance.get("annotation_sha256")
            or _stable_json_sha256(annotation)
        ),
        "panel_evidence_sha256": panel_evidence.get("sha256"),
        "metric_eligible": item.get("metric_eligible"),
        "context_eligible": item.get("context_eligible"),
        "metric_evidence_state": item.get("metric_evidence_state"),
    }


def _final_model_answer_context_item(item: dict[str, Any]) -> dict[str, Any]:
    """Expose raw content only when the requested mode was attested.

    Ineligible rows still participate in coverage/provenance manifests, but
    their answer and derived evidence cannot become narrative evidence for a
    mode that the transport layer did not verify.
    """

    context = copy.deepcopy(item)
    requested_mode = str(context.get("mode") or "unknown")
    context_eligible = context.get(
        "context_eligible",
        context.get("metric_eligible", True),
    ) is not False
    context["requested_mode"] = requested_mode
    context["verified_mode"] = requested_mode if context_eligible else None
    context["context_access"] = (
        "full_text" if context_eligible else "metadata_only"
    )
    if context_eligible:
        return context
    panel_evidence = (
        context.get("panel_evidence")
        if isinstance(context.get("panel_evidence"), dict)
        else {}
    )
    context.pop("answer_text", None)
    context.pop("annotation", None)
    context.pop("citations", None)
    context["content_withheld_reason"] = str(
        panel_evidence.get("reason") or "metric_ineligible"
    )
    return context


def _select_final_answer_context(
    answers: list[dict[str, Any]],
    *,
    corpus_manifest: dict[str, Any],
    max_answers: int = FINAL_CONTEXT_MAX_ANSWERS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select the smallest exact cover of observed evidence strata.

    A rarest-feature branch-and-bound search examines the complete feasible
    set-cover space while memoizing equivalent states. Unlike a greedy cover,
    it cannot reject a valid selection merely because an early locally useful
    answer consumed one of the limited slots. Eligible rows retain full raw
    text; ineligible rows are reduced to provenance-only context after
    selection so an unattested requested mode cannot influence the narrative.
    """

    if not answers:
        raise OpenRouterError("Final answer context is empty")
    if not isinstance(max_answers, int) or max_answers < 1:
        raise OpenRouterError("Final answer context limit is invalid")

    ordered_answers = sorted(
        answers,
        key=lambda item: (
            int(item.get("scenario_sequence") or 0),
            str(item.get("provider_key") or ""),
            str(item.get("mode") or ""),
            int(item.get("answer_id") or 0),
        ),
    )
    indexed = list(enumerate(ordered_answers))
    all_features = set().union(
        *(_selection_features(item) for _, item in indexed)
    )
    ordered_features = sorted(all_features)
    feature_bits = {
        feature: 1 << index
        for index, feature in enumerate(ordered_features)
    }
    all_features_mask = (1 << len(ordered_features)) - 1
    item_masks = {
        index: sum(
            feature_bits[feature]
            for feature in _selection_features(item)
        )
        for index, item in indexed
    }
    item_serialized_bytes = {
        index: len(json.dumps(item, ensure_ascii=False).encode("utf-8"))
        for index, item in indexed
    }

    provider_mode_count = len(
        {
            _selection_dimension_values(item)["provider_mode"]
            for item in ordered_answers
        }
    )
    if provider_mode_count > max_answers:
        raise OpenRouterError(
            "Final answer selection cannot represent every provider/mode "
            f"within {max_answers} full answers"
        )

    def selection_signature(indexes: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(
            int(ordered_answers[index].get("answer_id") or 0)
            for index in indexes
        )

    # Equal or strictly dominated masks can never improve coverage, byte cost
    # or cardinality, so discard them before the exact search.
    by_mask: dict[int, tuple[int, int]] = {}
    for index, _item in indexed:
        mask = item_masks[index]
        size = item_serialized_bytes[index]
        current = by_mask.get(mask)
        if current is None or (
            size,
            selection_signature((index,)),
        ) < (
            current[0],
            selection_signature((current[1],)),
        ):
            by_mask[mask] = (size, index)
    candidates = [
        (mask, size, index)
        for mask, (size, index) in sorted(
            by_mask.items(),
            key=lambda item: selection_signature((item[1][1],)),
        )
    ]
    candidates = [
        candidate
        for candidate in candidates
        if not any(
            candidate[0] != other[0]
            and candidate[0] | other[0] == other[0]
            and candidate[1] >= other[1]
            for other in candidates
        )
    ]
    candidates_by_feature: dict[int, list[int]] = defaultdict(list)
    for candidate_index, (mask, _size, _index) in enumerate(candidates):
        for feature_index in range(len(ordered_features)):
            if mask & (1 << feature_index):
                candidates_by_feature[feature_index].append(candidate_index)

    def candidate_order(
        candidate_index: int,
        missing_mask: int,
    ) -> tuple[Any, ...]:
        mask, size, index = candidates[candidate_index]
        gain = (mask & missing_mask).bit_count()
        return (
            size / max(gain, 1),
            size,
            -gain,
            selection_signature((index,)),
        )

    # A deterministic greedy result supplies an early upper bound only. The
    # recursive search below remains authoritative and explores every cover
    # that could beat it.
    greedy_mask = 0
    greedy_cost = 0
    greedy_selected: tuple[int, ...] = ()
    while greedy_mask != all_features_mask and len(greedy_selected) < max_answers:
        missing_mask = all_features_mask & ~greedy_mask
        usable = [
            candidate_index
            for candidate_index, (mask, _size, _index) in enumerate(candidates)
            if mask & missing_mask
        ]
        if not usable:
            break
        selected_candidate = min(
            usable,
            key=lambda value: candidate_order(value, missing_mask),
        )
        mask, size, index = candidates[selected_candidate]
        greedy_mask |= mask
        greedy_cost += size
        greedy_selected = (*greedy_selected, index)

    best_key: tuple[int, int, tuple[int, ...]] = (
        (
            greedy_cost
            if greedy_mask == all_features_mask
            else 10**30
        ),
        len(greedy_selected),
        selection_signature(tuple(sorted(greedy_selected))),
    )
    best_selected: tuple[int, ...] = (
        tuple(sorted(greedy_selected))
        if greedy_mask == all_features_mask
        else ()
    )
    visited_cost: dict[tuple[int, int], int] = {}

    def search(
        covered_mask: int,
        serialized_bytes: int,
        selected: tuple[int, ...],
    ) -> None:
        nonlocal best_key, best_selected
        if serialized_bytes > best_key[0]:
            return
        if covered_mask == all_features_mask:
            normalized = tuple(sorted(selected))
            key = (
                serialized_bytes,
                len(normalized),
                selection_signature(normalized),
            )
            if key < best_key:
                best_key = key
                best_selected = normalized
            return
        if len(selected) >= max_answers:
            return

        state = (covered_mask, len(selected))
        if visited_cost.get(state, 10**30) <= serialized_bytes:
            return
        visited_cost[state] = serialized_bytes

        missing_mask = all_features_mask & ~covered_mask
        max_gain = max(
            ((mask & missing_mask).bit_count() for mask, _size, _index in candidates),
            default=0,
        )
        if max_gain < 1:
            return
        minimum_remaining = math.ceil(missing_mask.bit_count() / max_gain)
        missing_by_dimension = Counter(
            ordered_features[index][0]
            for index in range(len(ordered_features))
            if missing_mask & (1 << index)
        )
        if missing_by_dimension:
            minimum_remaining = max(
                minimum_remaining,
                max(missing_by_dimension.values()),
            )
        if len(selected) + minimum_remaining > max_answers:
            return

        missing_features = [
            index
            for index in range(len(ordered_features))
            if missing_mask & (1 << index)
        ]
        rarest_feature = min(
            missing_features,
            key=lambda index: (
                len(candidates_by_feature[index]),
                ordered_features[index],
            ),
        )
        options = sorted(
            candidates_by_feature[rarest_feature],
            key=lambda value: candidate_order(value, missing_mask),
        )
        for candidate_index in options:
            mask, size, index = candidates[candidate_index]
            if not mask & missing_mask:
                continue
            search(
                covered_mask | mask,
                serialized_bytes + size,
                (*selected, index),
            )

    search(0, 0, ())
    if not best_selected:
        raise OpenRouterError(
            "Final answer selection cannot cover required evidence strata "
            f"within {max_answers} full answers: {sorted(all_features)}"
        )
    selected_indexes = best_selected
    selected_index_set = set(selected_indexes)
    selected_answers = sorted(
        (ordered_answers[index] for index in selected_indexes),
        key=lambda item: (
            int(item.get("scenario_sequence") or 0),
            str(item.get("provider_key") or ""),
            str(item.get("mode") or ""),
            int(item.get("answer_id") or 0),
        ),
    )
    selected_features = set().union(
        *(_selection_features(item) for item in selected_answers)
    )
    coverage_complete = selected_features == all_features
    if not coverage_complete:
        missing = sorted(all_features - selected_features)
        raise OpenRouterError(
            "Final answer selection cannot cover required evidence strata "
            f"within {max_answers} full answers: {missing}"
        )

    selected_context = [
        _final_model_answer_context_item(item) for item in selected_answers
    ]
    selected_full_text_count = sum(
        item.get("context_access") == "full_text"
        for item in selected_context
    )

    omitted = sorted(
        (
            _selection_item_signature(item)
            for index, item in indexed
            if index not in selected_index_set
        ),
        key=lambda item: (
            int(item.get("prompt_id") or 0),
            str(item.get("provider_key") or ""),
            str(item.get("mode") or ""),
            int(item.get("answer_id") or 0),
        ),
    )
    manifest_core = {
        "version": FINAL_CONTEXT_SELECTION_VERSION,
        "full_corpus_digest": corpus_manifest.get("digest"),
        "full_corpus_critic_rows_sha256": corpus_manifest.get(
            "critic_rows_sha256"
        ),
        "max_answers": max_answers,
        "selected_count": len(selected_answers),
        "selected_full_text_count": selected_full_text_count,
        "selected_metadata_only_count": (
            len(selected_context) - selected_full_text_count
        ),
        "omitted_count": len(answers) - len(selected_answers),
        "selected_cells": [
            _selection_item_signature(item) for item in selected_answers
        ],
        "omitted_cells_sha256": _stable_json_sha256(omitted),
        "observed_coverage": _selection_coverage(answers),
        "selected_coverage": _selection_coverage(selected_answers),
        "coverage_complete": coverage_complete,
        "selected_serialized_utf8_bytes": len(
            json.dumps(selected_context, ensure_ascii=False).encode("utf-8")
        ),
    }
    return selected_context, {
        **manifest_core,
        "digest": _stable_json_sha256(manifest_core),
    }


async def _full_answer_context(
    run_id: str,
    *,
    critic_gate: dict[str, Any],
    critic_rows: list[dict[str, Any]],
    expected_corpus_cells: list[dict[str, Any]],
) -> dict[str, Any]:
    """Load the complete corpus and prove it is the critic-approved corpus."""

    legacy_contract = await _legacy_panel_run_contract(run_id)
    async with SessionLocal() as session:
        rows = (
            await session.execute(
                select(ModelAnswer, VisibilityPrompt, AnswerAnnotation)
                .join(VisibilityPrompt, ModelAnswer.prompt_id == VisibilityPrompt.id)
                .outerjoin(
                    AnswerAnnotation,
                    AnswerAnnotation.answer_id == ModelAnswer.id,
                )
                .where(ModelAnswer.run_id == run_id)
                .order_by(
                    VisibilityPrompt.sequence,
                    ModelAnswer.mode,
                    ModelAnswer.provider_key,
                    ModelAnswer.id,
                )
            )
        ).all()
    model_rows = list(rows)
    current_rows = _rows_from_full_answer_models(
        model_rows,
        legacy_contract=legacy_contract,
    )
    current_manifest = _final_corpus_manifest(
        current_rows,
        expected_cells=expected_corpus_cells,
    )
    gate_manifest = critic_gate.get("corpus_manifest")
    gate_rows_sha256 = (
        (critic_gate.get("provenance") or {}).get("rows_sha256")
        if isinstance(critic_gate.get("provenance"), dict)
        else None
    )
    mismatch_reasons: list[str] = []
    if critic_gate.get("passed") is not True:
        mismatch_reasons.append("critic_gate_not_passed")
    if not isinstance(gate_manifest, dict):
        mismatch_reasons.append("critic_corpus_manifest_missing")
    elif gate_manifest.get("digest") != current_manifest["digest"]:
        mismatch_reasons.append("critic_corpus_manifest_mismatch")
    if gate_rows_sha256 != current_manifest["critic_rows_sha256"]:
        mismatch_reasons.append("critic_rows_digest_mismatch")
    if (
        _stable_json_sha256(_critic_row_provenance(critic_rows))
        != current_manifest["critic_rows_sha256"]
    ):
        mismatch_reasons.append("in_memory_critic_rows_mismatch")
    if not current_manifest["complete"]:
        mismatch_reasons.append("corpus_incomplete")
    if mismatch_reasons:
        raise OpenRouterError(
            "Final answer corpus failed closed: "
            + ", ".join(mismatch_reasons)
        )

    return {
        "manifest": current_manifest,
        "answers": _full_answer_corpus_items(
            model_rows,
            legacy_contract=legacy_contract,
        ),
    }


def _build_public_report(
    *,
    profile: dict[str, Any],
    technical: dict[str, Any],
    technical_review: dict[str, Any],
    metrics: dict[str, Any],
    canonical_intent_taxonomy: bool = False,
) -> dict[str, Any]:
    coverage = technical.get("coverage")
    if not isinstance(coverage, dict):
        evaluated = _non_negative_int(technical.get("evaluated_pages")) or 0
        coverage = _technical_page_coverage(
            evaluated,
            {
                "discovered_count": technical.get("discovered_pages"),
                "coverage_state": technical.get("coverage_state"),
            },
        )
    coverage_state = str(coverage.get("coverage_state") or "unknown")
    metric_state = (
        coverage_state
        if coverage_state in {"limited", "unknown"}
        else technical.get("state")
    )
    memory_slices = (
        (metrics.get("parent_discovery") or {}).get("memory") or {},
        (metrics.get("portfolio_visibility") or {}).get("memory") or {},
        (metrics.get("brand_knowledge") or {}).get("memory") or {},
    )
    memory_available = any(
        item.get("data_state") in {"complete", "limited"}
        and int(item.get("valid_answers") or 0) > 0
        for item in memory_slices
        if isinstance(item, dict)
    )
    memory_observational = any(
        item.get("evidence_state") == "legacy_observational"
        or int(item.get("observational_answers") or 0) > 0
        for item in memory_slices
        if isinstance(item, dict)
    )
    brand_knowledge_public = copy.deepcopy(metrics["brand_knowledge"])
    providers_public = copy.deepcopy(metrics["providers"])
    if memory_observational:
        def withhold_observational_facets(slice_data: Any) -> None:
            if not isinstance(slice_data, dict):
                return
            slice_data.pop("facets", None)
            slice_data["qualitative_context_withheld"] = True

        withhold_observational_facets(brand_knowledge_public.get("memory"))
        for provider in brand_knowledge_public.get("providers") or []:
            if isinstance(provider, dict):
                withhold_observational_facets(provider.get("memory"))
        for provider in providers_public:
            if not isinstance(provider, dict):
                continue
            provider_knowledge = provider.get("brand_knowledge")
            if isinstance(provider_knowledge, dict):
                withhold_observational_facets(
                    provider_knowledge.get("memory")
                )
    portfolio_scope = metrics.get("portfolio_scope")
    if not isinstance(portfolio_scope, dict):
        portfolio_scope = {}
    portfolio_web = metrics["portfolio_visibility"]["web"]
    portfolio_scope_unavailable = (
        portfolio_scope.get("state") == "unavailable"
        or portfolio_web.get("data_state") == "unavailable"
    )
    portfolio_scope_limit = (
        "Состав продуктового портфеля не подтверждён данными сайта: "
        "продуктовые показатели не рассчитаны и не заменены нулём."
        if portfolio_scope_unavailable
        else None
    )
    model_consistency_value = metrics.get("model_consistency")
    model_consistency_available = isinstance(
        model_consistency_value,
        (int, float),
    )
    model_consistency_reason = (
        "target_portfolio_unconfirmed"
        if portfolio_scope_unavailable
        else (
            None
            if model_consistency_available
            else "insufficient_valid_provider_data"
        )
    )
    model_consistency_public = {
        "value": model_consistency_value,
        "data_state": (
            "complete" if model_consistency_available else "unavailable"
        ),
        "state": (
            "available" if model_consistency_available else "unknown"
        ),
        "unavailable_reason": model_consistency_reason,
    }
    methodology_modes = [
        {
            "name": "Безбрендовые сценарии · с веб-поиском",
            "description": (
                "Показывают, назовут ли системы материнский бренд без "
                "подсказки, опираясь на актуальные источники."
                if portfolio_scope_unavailable
                else (
                    "Показывают, назовут ли системы бренд или его "
                    "предложение без подсказки, опираясь на актуальные "
                    "источники."
                )
            ),
        },
    ]
    if memory_available:
        methodology_modes.append(
            {
                "name": (
                    "Безбрендовые сценарии · без зафиксированного веб-поиска"
                    if memory_observational
                    else "Безбрендовые сценарии · без веб-поиска"
                ),
                "description": (
                    "Исторический наблюдательный срез: модели вызваны без "
                    "онлайн-варианта и не вернули ссылок или сигналов "
                    "веб-инструментов, но явное отключение веб-доступа не "
                    "было сохранено в транспортном контракте."
                    if memory_observational
                    else (
                        "Показывают обнаружение по знаниям, уже "
                        "закрепившимся в моделях."
                    )
                ),
            }
        )
    methodology_modes.append(
        {
            "name": "Бренд назван прямо",
            "description": (
                "Отдельно оценивает, насколько конкретно и по существу "
                "модели способны рассказать о бренде."
            ),
        }
    )
    return {
        "version": "2026-07-v3",
        "brand": {
            "name": profile.get("brand_name") or "Бренд не определён",
            "site_type": profile.get("site_type"),
            "category": profile.get("category"),
            "products": profile.get("products") or [],
            "positioning": profile.get("positioning"),
            "confidence": profile.get("confidence"),
            "entity_scope": profile.get("entity_scope") or [],
        },
        "key_metrics": {
            "technical_access": {
                "value": technical.get("score"),
                "unit": "/ 100",
                "label": "Техническая готовность проверенного среза",
                "state": metric_state,
                "access_state": technical.get("state"),
                "evaluated_pages": coverage.get("evaluated_pages"),
                "discovered_pages": coverage.get("discovered_pages"),
                "coverage_rate": coverage.get("coverage_rate"),
                "coverage_state": coverage_state,
                "coverage_label": coverage.get("coverage_label"),
                "scope_label": coverage.get("scope_label"),
            },
            "parent_discovery": {
                "value": metrics["parent_discovery"]["web"].get("score"),
                "unit": "/ 100",
                "label": "Индекс видимости бренда",
                "state": metrics["parent_discovery"]["web"].get("state"),
            },
            "portfolio_capture": {
                "value": portfolio_web.get("score"),
                "unit": "/ 100",
                "label": "Индекс видимости продуктов",
                "state": (
                    "unavailable"
                    if portfolio_scope_unavailable
                    else portfolio_web.get("state")
                ),
                "unavailable_reason": portfolio_web.get(
                    "unavailable_reason"
                ),
                "data_state": portfolio_web.get("data_state"),
            },
            "brand_knowledge": {
                "value": metrics["brand_knowledge"]["memory"].get("specific_rate"),
                "unit": "%",
                "label": (
                    "Конкретика без зафиксированного веб-поиска"
                    if memory_observational
                    else "Конкретные знания без веба"
                ),
                "state": metrics["brand_knowledge"]["memory"].get("state"),
                "data_state": metrics["brand_knowledge"]["memory"].get(
                    "data_state"
                ),
                "evidence_state": metrics["brand_knowledge"]["memory"].get(
                    "evidence_state"
                ),
                "limitation_reason": metrics["brand_knowledge"]["memory"].get(
                    "limitation_reason"
                ),
            },
        },
        "technical": {
            **technical,
            "review": technical_review,
        },
        "discovery": {
            "parent": metrics["parent_discovery"],
            "portfolio": metrics["portfolio_visibility"],
            "portfolio_scope": portfolio_scope,
            "paired_web_lift": metrics["paired_web_lift"],
            "model_consistency": model_consistency_public,
            "providers": providers_public,
            "intents": metrics["intents"],
            "sentiment": metrics["sentiment"],
        },
        "brand_knowledge": brand_knowledge_public,
        "portfolio_scope": portfolio_scope,
        "visibility": {
            # Compatibility for reports rendered by an older frontend.
            "web": metrics["web"],
            "memory": metrics["memory"],
            "model_consistency": model_consistency_public,
            "providers": providers_public,
            "intents": metrics["intents"],
            "sentiment": metrics["sentiment"],
        },
        "competitors": metrics["competitors"],
        "data_quality": metrics["quality"],
        "methodology": {
            "intent_taxonomy_version": (
                "canonical-v1" if canonical_intent_taxonomy else "legacy-v1"
            ),
            "intent_definitions": (
                INTENT_DEFINITIONS if canonical_intent_taxonomy else {}
            ),
            "summary": (
                "Сервис отделяет безбрендовое обнаружение от брендовой диагностики, "
                + (
                    (
                        "сопоставляет веб-срез с историческим наблюдательным "
                        "срезом без зафиксированного веб-поиска, а затем "
                        "программно "
                    )
                    if memory_observational
                    else "сравнивает ответы с веб-поиском и без него, а затем программно "
                    if memory_available
                    else "анализирует подтверждённые ответы с веб-поиском и программно "
                )
                + "считает показатели из доказательной разметки."
            ),
            "panel_evidence": metrics["quality"].get("panel_evidence") or {},
            "modes": methodology_modes,
            "note": metrics["metric_note"],
            "offline_limit": (
                (
                    (
                        f"{CANONICAL_OBSERVATIONAL_MEMORY_LIMITATION} "
                        "Perplexity в этот срез не входит."
                        if memory_observational
                        else (
                            "Perplexity не входит в сравнение памяти: её "
                            "доступный поисковый режим нельзя честно "
                            "отключить."
                        )
                    )
                )
                if memory_available
                else (
                    "Сохранённый срез без веб-поиска не участвует в процентах: "
                    "для него нельзя подтвердить, что внешние источники были "
                    "технически отключены."
                )
            ),
            "portfolio_scope_limit": portfolio_scope_limit,
        },
        "data_quality": metrics["quality"],
        "illustrations": [],
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# {str(report.get('headline') or '').strip()}",
        "",
        f"**Вердикт.** {str(report.get('verdict') or '').strip()}",
        "",
        str(report.get("executive_summary") or "").strip(),
    ]
    for section in report.get("sections") or []:
        lines.extend(
            [
                "",
                f"## {str(section.get('heading') or '').strip()}",
                "",
                str(section.get("body") or "").strip(),
            ]
        )
    lines.extend(["", "## Что изменить в первую очередь", ""])
    priority_labels = {
        "now": "Сейчас",
        "next": "Следом",
        "later": "После основных исправлений",
    }
    for action in report.get("actions") or []:
        lines.extend(
            [
                f"### {priority_labels.get(action.get('priority'), 'Действие')}: "
                f"{str(action.get('title') or '').strip()}",
                "",
                str(action.get("why") or "").strip(),
                "",
                f"**Шаг.** {str(action.get('step') or '').strip()}",
                "",
                f"**Основание.** {str(action.get('evidence') or '').strip()}",
                "",
            ]
        )
    lines.extend(["## Где заканчивается точность экспресс-снимка", ""])
    for limitation in report.get("limitations") or []:
        lines.append(f"- {str(limitation).strip()}")
    return "\n".join(lines).strip()


def _validate_final_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    sections = report.get("sections")
    actions = report.get("actions")
    if "illustrations" in report:
        errors.append(
            "Концепции иллюстраций не должны входить в аналитический отчёт."
        )
    if not isinstance(sections, list) or not sections:
        errors.append("В отчёте должен быть хотя бы один содержательный раздел.")
    if not isinstance(actions, list) or not actions:
        errors.append("В отчёте должно быть хотя бы одно приоритетное действие.")
    return errors


def _final_report_semantic_review_payload(
    public_report: dict[str, Any],
    candidate_report: dict[str, Any],
    *,
    selected_answer_context: list[dict[str, Any]],
    answer_selection_manifest: dict[str, Any],
) -> dict[str, Any]:
    evidence_document = {
        "report_data": public_report,
        "selected_answer_context": selected_answer_context,
        "answer_selection_manifest": answer_selection_manifest,
    }
    availability_contract = metric_availability_contract(public_report)
    for item in availability_contract:
        path = str(item.get("path") or "/")
        item["path"] = (
            "/report_data"
            if path == "/"
            else f"/report_data{path}"
        )
    return {
        "evidence_document": evidence_document,
        "metric_availability_contract": availability_contract,
        "candidate_report": candidate_report,
        "deterministic_precheck_errors": (
            deterministic_report_semantic_errors(
                candidate_report,
                public_report,
            )
        ),
    }


async def _final_report_semantic_review_artifact(
    run_id: str,
    *,
    public_report: dict[str, Any],
    candidate_report: dict[str, Any],
    selected_answer_context: list[dict[str, Any]],
    answer_selection_manifest: dict[str, Any],
    attempt: int,
) -> dict[str, Any]:
    """Run and persist one independent, fail-closed semantic review."""

    if not 1 <= attempt <= MAX_FINAL_REPORT_REPAIRS + 1:
        raise ValueError("Final report semantic review is outside the bounded loop")
    artifact_key = f"final_report_semantic_gate_a{attempt}"
    review_input = _final_report_semantic_review_payload(
        public_report,
        candidate_report,
        selected_answer_context=selected_answer_context,
        answer_selection_manifest=answer_selection_manifest,
    )
    cached = await _artifact_output(
        run_id,
        artifact_key,
        input_json=review_input,
        model=REPORT_SEMANTIC_MODEL,
        prompt_version=REPORT_SEMANTIC_GATE_VERSION,
    )
    if isinstance(cached, dict):
        cached = normalize_report_semantic_review(
            cached,
            evidence_document=review_input["evidence_document"],
            candidate_report=candidate_report,
            report_data=public_report,
        )
        cached_errors = validate_report_semantic_review(
            cached,
            evidence_document=review_input["evidence_document"],
            candidate_report=candidate_report,
        )
        if cached_errors:
            raise OpenRouterError(
                "Cached final report semantic review is invalid: "
                + "; ".join(cached_errors)
            )
        return cached

    await _save_artifact(
        run_id,
        stage_key="report",
        artifact_key=artifact_key,
        status="running",
        model=REPORT_SEMANTIC_MODEL,
        input_json=review_input,
        prompt_version=REPORT_SEMANTIC_GATE_VERSION,
    )
    review: dict[str, Any] | None = None
    raw_text: str | None = None
    usage: dict[str, Any] | None = None
    try:
        review, raw_text, usage = await review_final_report_semantics(
            review_input,
            attempt=attempt,
        )
        review = normalize_report_semantic_review(
            review,
            evidence_document=review_input["evidence_document"],
            candidate_report=candidate_report,
            report_data=public_report,
        )
        review_errors = validate_report_semantic_review(
            review,
            evidence_document=review_input["evidence_document"],
            candidate_report=candidate_report,
        )
        if review_errors:
            raise OpenRouterError(
                "Final report semantic review is invalid: "
                + "; ".join(review_errors)
            )
        await _save_artifact(
            run_id,
            stage_key="report",
            artifact_key=artifact_key,
            status="completed",
            model=REPORT_SEMANTIC_MODEL,
            input_json=review_input,
            output_json=review,
            raw_text=raw_text,
            usage_json=usage,
            prompt_version=REPORT_SEMANTIC_GATE_VERSION,
        )
        return review
    except Exception as exc:
        await _save_artifact(
            run_id,
            stage_key="report",
            artifact_key=artifact_key,
            status="failed",
            model=REPORT_SEMANTIC_MODEL,
            input_json=review_input,
            output_json=review,
            raw_text=raw_text,
            usage_json=usage,
            error_message=str(exc),
            prompt_version=REPORT_SEMANTIC_GATE_VERSION,
        )
        raise


def _sanitize_headline_emphasis(report: dict[str, Any]) -> dict[str, Any]:
    headline = str(report.get("headline") or "")
    raw = report.get("headline_emphasis")
    accepted: list[str] = []
    occupied: list[tuple[int, int]] = []
    if isinstance(raw, list):
        for value in raw[:2]:
            phrase = str(value or "").strip()
            if not phrase:
                continue
            start = headline.find(phrase)
            if start < 0:
                continue
            span = (start, start + len(phrase))
            if any(span[0] < other[1] and other[0] < span[1] for other in occupied):
                continue
            occupied.append(span)
            accepted.append(phrase)
    report["headline_emphasis"] = accepted
    return report


def _resolve_json_pointer(document: Any, pointer: str) -> Any:
    if not pointer.startswith("/"):
        raise KeyError(pointer)
    value = document
    for raw_part in pointer.split("/")[1:]:
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(value, dict) and part in value:
            value = value[part]
        elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        else:
            raise KeyError(pointer)
    return value


_ILLUSTRATION_NUMBER_LITERAL = re.compile(r"(?<![\w])\d+(?:[.,]\d+)?")


def _collect_report_numbers(value: Any) -> set[float]:
    numbers: set[float] = set()
    if isinstance(value, bool) or value is None:
        return numbers
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            numbers.add(float(value))
        return numbers
    if isinstance(value, str):
        for match in _ILLUSTRATION_NUMBER_LITERAL.finditer(value):
            numbers.add(float(match.group(0).replace(",", ".")))
        return numbers
    if isinstance(value, dict):
        for child in value.values():
            numbers.update(_collect_report_numbers(child))
    elif isinstance(value, list):
        for child in value:
            numbers.update(_collect_report_numbers(child))
    return numbers


def _illustration_numeric_claim_errors(
    concept: dict[str, Any],
    public_report: dict[str, Any],
) -> list[str]:
    allowed: set[float] = set()
    for path in concept.get("evidence_paths") or []:
        if not isinstance(path, str):
            continue
        try:
            allowed.update(
                _collect_report_numbers(_resolve_json_pointer(public_report, path))
            )
        except KeyError:
            continue
    errors: list[str] = []
    for field in ("title", "caption", "core_claim", "alt_text"):
        text = str(concept.get(field) or "")
        if field == "title" and re.fullmatch(
            r"\s*(?:Схема|Иллюстрация|Рисунок)\s+\d+\s*",
            text,
            flags=re.IGNORECASE,
        ):
            continue
        for match in _ILLUSTRATION_NUMBER_LITERAL.finditer(text):
            observed = float(match.group(0).replace(",", "."))
            if any(abs(observed - expected) <= 0.051 for expected in allowed):
                continue
            errors.append(
                f"{field}: число {match.group(0)} не подтверждено "
                "указанными evidence_paths."
            )
    return errors


def _normalize_unpaired_memory_illustration(
    concepts: dict[str, Any],
    public_report: dict[str, Any],
) -> dict[str, Any]:
    """Keep visual copy honest when no attested web/memory pairs exist."""

    discovery = public_report.get("discovery")
    paired = (
        discovery.get("paired_web_lift")
        if isinstance(discovery, dict)
        and isinstance(discovery.get("paired_web_lift"), dict)
        else {}
    )
    if _non_negative_int(paired.get("n_pairs")) != 0:
        return concepts
    normalized = copy.deepcopy(concepts)
    evidence_candidates = (
        "/discovery/paired_web_lift/n_pairs",
        "/discovery/parent/memory/data_state",
        "/discovery/portfolio/memory/data_state",
        "/brand_knowledge/memory/data_state",
    )
    evidence_paths: list[str] = []
    for path in evidence_candidates:
        try:
            _resolve_json_pointer(public_report, path)
        except KeyError:
            continue
        evidence_paths.append(path)
    for item in normalized.get("illustrations") or []:
        if not isinstance(item, dict) or item.get("role") != "web_memory_gap":
            continue
        item.update(
            {
                "title": "Сопоставимого среза для сравнения пока нет",
                "caption": (
                    "В этом замере нет достаточного набора подтверждённых пар "
                    "«с веб-поиском / без веб-поиска», поэтому направление и "
                    "величину разницы оценить нельзя."
                ),
                "alt_text": (
                    "Схема показывает, что сопоставимых подтверждённых данных "
                    "для сравнения двух режимов пока недостаточно."
                ),
                "core_claim": (
                    "Разницу между веб-поиском и знаниями моделей нельзя "
                    "оценить по этому замеру."
                ),
                "evidence_paths": evidence_paths,
            }
        )
    return normalized


def _validate_illustration_concepts(
    concepts: dict[str, Any],
    public_report: dict[str, Any] | None = None,
) -> list[str]:
    errors: list[str] = []
    illustrations = concepts.get("illustrations")
    if not isinstance(illustrations, list) or len(illustrations) != 3:
        errors.append("Нужно ровно три концепции иллюстраций.")
        return errors
    expected_roles = [
        "technical_access",
        "competitive_visibility",
        "web_memory_gap",
    ]
    actual_roles = [
        item.get("role") if isinstance(item, dict) else None
        for item in illustrations
    ]
    if actual_roles != expected_roles:
        errors.append(
            "Концепции должны идти в порядке: техническая доступность, "
            "конкурентная видимость, веб-поиск и память моделей."
        )
    required_text = (
        "title",
        "caption",
        "alt_text",
        "core_claim",
        "context_for_image",
    )
    allowed_roots = {
        "technical_access": ("/technical/", "/key_metrics/technical_access"),
        "competitive_visibility": (
            "/discovery/",
            "/competitors/",
            "/brand/",
            "/key_metrics/parent_discovery",
            "/key_metrics/portfolio_capture",
        ),
        "web_memory_gap": (
            "/discovery/",
            "/brand_knowledge/",
            "/methodology/",
            "/key_metrics/brand_knowledge",
        ),
    }
    visual_theses: set[str] = set()
    for sequence, item in enumerate(illustrations, start=1):
        if not isinstance(item, dict) or any(
            not isinstance(item.get(field), str) or not item[field].strip()
            for field in required_text
        ):
            errors.append(
                f"В концепции № {sequence} должны быть заполнены все текстовые поля."
            )
            continue
        if public_report is not None:
            semantic_errors = deterministic_report_semantic_errors(
                {
                    "headline": item["title"],
                    "verdict": item["core_claim"],
                    "executive_summary": item["caption"],
                },
                public_report,
                enforce_report_contract=False,
            )
            errors.extend(
                f"Концепция № {sequence}: {error}"
                for error in semantic_errors
            )
        paths = item.get("evidence_paths")
        if not isinstance(paths, list) or not paths:
            errors.append(
                f"Концепция № {sequence} не содержит доказательных JSON-путей."
            )
        role = str(item.get("role") or "")
        for path in paths or []:
            roots = allowed_roots.get(role, ())
            path_allowed = isinstance(path, str) and any(
                path == root.rstrip("/") or path.startswith(root)
                for root in roots
            )
            if not path_allowed:
                errors.append(
                    f"Концепция № {sequence} ссылается на недопустимый факт: {path}."
                )
                continue
            if public_report is not None:
                try:
                    _resolve_json_pointer(public_report, path)
                except KeyError:
                    errors.append(
                        f"Концепция № {sequence} ссылается на отсутствующий факт: {path}."
                    )
        if public_report is not None:
            errors.extend(
                f"Концепция № {sequence}: {error}"
                for error in _illustration_numeric_claim_errors(
                    item,
                    public_report,
                )
            )
        brief = item.get("creative_brief")
        if not isinstance(brief, dict) or any(
            not isinstance(brief.get(field), str) or not brief[field].strip()
            for field in (
                "visual_thesis",
                "scene",
                "composition",
                "materials_and_light",
                "emotional_tone",
                "target_treatment",
                "diversity_move",
            )
        ):
            errors.append(
                f"В концепции № {sequence} не заполнен творческий бриф."
            )
        elif str(brief["visual_thesis"]).casefold().strip() in visual_theses:
            errors.append(
                f"Концепция № {sequence} повторяет визуальную идею другой иллюстрации."
            )
        else:
            visual_theses.add(str(brief["visual_thesis"]).casefold().strip())
    return errors


def _final_report_payload(
    public_report: dict[str, Any],
    answer_corpus: dict[str, Any],
) -> dict[str, Any]:
    selected_answers, selection_manifest = _select_final_answer_context(
        answer_corpus["answers"],
        corpus_manifest=answer_corpus["manifest"],
    )
    return {
        "report_data": public_report,
        "answer_corpus_manifest": answer_corpus["manifest"],
        "answer_selection_manifest": selection_manifest,
        "selected_full_answers": selected_answers,
    }


def _final_input_preflight(
    payload: dict[str, Any],
    *,
    token_budget: int | None = None,
    reserve_tokens: int = 0,
) -> dict[str, Any]:
    """Conservatively estimate the complete user payload before the API call."""

    serialized = json.dumps(payload, ensure_ascii=False)
    serialized_bytes = len(serialized.encode("utf-8"))
    estimated_tokens = math.ceil(serialized_bytes / 3)
    budget = (
        settings.FINAL_INPUT_TOKEN_BUDGET
        if token_budget is None
        else token_budget
    )
    accepted = (
        isinstance(budget, int)
        and budget > 0
        and isinstance(reserve_tokens, int)
        and reserve_tokens >= 0
        and estimated_tokens + reserve_tokens <= budget
    )
    selection_manifest = payload.get("answer_selection_manifest")
    if not isinstance(selection_manifest, dict):
        selection_manifest = {}
    return {
        "state": "complete" if accepted else "limited",
        "accepted": accepted,
        "payload_sha256": hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest(),
        "serialized_utf8_bytes": serialized_bytes,
        "estimated_input_tokens": estimated_tokens,
        "input_token_budget": budget,
        "repair_reserve_tokens": reserve_tokens,
        "estimated_headroom_tokens": (
            budget - estimated_tokens - reserve_tokens
            if isinstance(budget, int) and budget > 0
            else None
        ),
        "selection_digest": selection_manifest.get("digest"),
        "selected_answer_count": selection_manifest.get("selected_count"),
        "selected_full_text_count": selection_manifest.get(
            "selected_full_text_count"
        ),
        "selected_metadata_only_count": selection_manifest.get(
            "selected_metadata_only_count"
        ),
        "selection_coverage_complete": selection_manifest.get(
            "coverage_complete"
        ),
        "estimation_contract": "ceil(serialized_utf8_bytes / 3)",
    }


async def _final_report_author_candidate(
    run_id: str,
    *,
    system: str,
    payload: dict[str, Any],
) -> tuple[dict[str, Any], str | None, dict[str, Any] | None]:
    """Return one structurally valid authored candidate.

    The author cache is intentionally separate from the publishable final
    report.  It is keyed by the complete author payload, model and final
    prompt version, and only a candidate that passes deterministic structural
    validation is stored as completed.  A malformed author response gets one
    bounded structural repair; a second malformed response fails closed.
    """

    cached = await _artifact_output(
        run_id,
        FINAL_REPORT_AUTHOR_ARTIFACT_KEY,
        input_json=payload,
        model=ANALYSIS_MODEL,
        prompt_version=FINAL_REPORT_VERSION,
    )
    if isinstance(cached, dict):
        cached = _sanitize_headline_emphasis(copy.deepcopy(cached))
        if not _validate_final_report(cached):
            return cached, None, None

    await _save_artifact(
        run_id,
        stage_key="report",
        artifact_key=FINAL_REPORT_AUTHOR_ARTIFACT_KEY,
        status="running",
        model=ANALYSIS_MODEL,
        input_json=payload,
        prompt_version=FINAL_REPORT_VERSION,
    )
    last_errors: list[str] = []
    rejected_report: dict[str, Any] | None = None
    last_raw_text: str | None = None
    last_usage: dict[str, Any] | None = None
    try:
        for attempt in range(MAX_FINAL_STRUCTURE_REPAIRS + 1):
            user_payload = dict(payload)
            if attempt:
                user_payload["structure_validation_errors_to_fix"] = (
                    last_errors
                )
                user_payload["rejected_report"] = rejected_report
                repair_preflight = _final_input_preflight(user_payload)
                if not repair_preflight["accepted"]:
                    raise OpenRouterError(
                        "Final report structure repair input exceeds the "
                        "configured context budget: "
                        f"{repair_preflight['estimated_input_tokens']} "
                        "estimated tokens > "
                        f"{repair_preflight['input_token_budget']}."
                    )
            result = await chat(
                model=ANALYSIS_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": json.dumps(
                            user_payload,
                            ensure_ascii=False,
                        ),
                    },
                ],
                response_schema=FINAL_REPORT_SCHEMA,
                schema_name="aiv_final_report",
                reasoning_effort="high",
                max_tokens=30_000,
                temperature=0.15,
            )
            last_raw_text = result.text
            last_usage = result.usage
            if not isinstance(result.parsed, dict):
                last_errors = ["Ответ не является объектом."]
                rejected_report = None
                continue
            candidate = _sanitize_headline_emphasis(result.parsed)
            last_errors = _validate_final_report(candidate)
            if last_errors:
                rejected_report = candidate
                continue
            await _save_artifact(
                run_id,
                stage_key="report",
                artifact_key=FINAL_REPORT_AUTHOR_ARTIFACT_KEY,
                status="completed",
                model=ANALYSIS_MODEL,
                input_json=payload,
                output_json=candidate,
                raw_text=last_raw_text,
                usage_json=last_usage,
                prompt_version=FINAL_REPORT_VERSION,
            )
            return candidate, last_raw_text, last_usage
        raise OpenRouterError(
            "; ".join(last_errors)
            or "Final report structural validation failed"
        )
    except Exception as exc:
        await _save_artifact(
            run_id,
            stage_key="report",
            artifact_key=FINAL_REPORT_AUTHOR_ARTIFACT_KEY,
            status="failed",
            model=ANALYSIS_MODEL,
            input_json=payload,
            raw_text=last_raw_text,
            usage_json=last_usage,
            error_message=str(exc),
            prompt_version=FINAL_REPORT_VERSION,
        )
        raise


async def _invalidate_blocked_final_report_author_candidate(
    run_id: str,
    *,
    payload: dict[str, Any],
    candidate: dict[str, Any],
    semantic_review: dict[str, Any],
    blockers: list[str],
) -> None:
    """Make a semantically blocked author candidate non-reusable."""

    candidate_sha256 = _stable_json_sha256(candidate)
    await _save_artifact(
        run_id,
        stage_key="report",
        artifact_key=FINAL_REPORT_AUTHOR_ARTIFACT_KEY,
        status="failed",
        model=ANALYSIS_MODEL,
        input_json=payload,
        output_json={
            "state": "semantic_block",
            "candidate_sha256": candidate_sha256,
            "semantic_verdict": semantic_review.get("verdict"),
            "blockers": blockers,
        },
        error_message=(
            "Final report author candidate was blocked by the semantic gate: "
            + "; ".join(blockers)
        ),
        prompt_version=FINAL_REPORT_VERSION,
    )


async def _final_report(
    run_id: str,
    public_report: dict[str, Any],
    answer_corpus: dict[str, Any],
) -> dict[str, Any]:
    system = f"""
Ты выпускающий аналитик сервиса AI Visibility команды RW+. Исследуемый бренд,
его продукты, рынок и конкурентов всегда бери только из переданного payload.
Для сайта rw.plus исследуемый бренд может совпадать с издателем сервиса — это
нормально. Собери итоговый
отчёт об AI visibility этого бренда. Работай только с переданными данными. Не
вычисляй новые показатели и не исправляй программно рассчитанные значения.

Структура:
- первая фраза даёт управленческий вывод;
- headline должен быть немного короче обычного газетного заголовка. В
  headline_emphasis верни одну или две точные непересекающиеся подстроки из
  headline, которые заслуживают более крупного кегля. Не перефразируй их;
- первый содержательный раздел — подробный технический аудит: доступ
  содержательных страниц, robots.txt, серверный HTML/CSR, Schema.org,
  формы входа, таблица страниц, доказательства и практические действия;
- каждый вывод технического аудита ограничивай проверенной выборкой:
  «на двух проверенных страницах», а не «сайт целиком»; число страниц бери
  из technical.coverage.evaluated_pages;
- если technical.coverage.coverage_state равно limited или unknown, прямо
  скажи, что оценка относится к проверенным страницам и не описывает весь
  сайт; unknown не называй полным охватом;
- техническая доступность и CSR/SSR разобраны отдельно от видимости в ответах;
- безбрендовое обнаружение материнского бренда, захват продуктового портфеля
  и знание бренда по прямому вопросу — три разные величины;
- в brand_knowledge поля answer_count и answer_rate означают ответы по
  существу, а не конкретику. Словами «конкретный», «конкретика» и
  «конкретно» описывай только specific_count и specific_rate. Всегда сверяй
  числитель, знаменатель и название метрики перед публикацией;
- ноль объясняй как наблюдение с явным числителем и знаменателем, а отсутствие
  или неполноту данных — как unknown/limited, никогда как 0;
- если discovery.portfolio_scope.state=unavailable либо
  discovery.portfolio.web.unavailable_reason=target_portfolio_unconfirmed,
  состав предложения не подтверждён самим сайтом: не утверждай, что продукты
  отсутствуют, не описывай их видимость как нулевую и не переноси сюда число
  материнского бренда. Не обсуждай продуктовый результат в headline, verdict,
  executive_summary, sections или actions. Добавь ровно один отдельный элемент
  limitations и дословно: «{CANONICAL_UNAVAILABLE_PORTFOLIO_LIMITATION}»;
- если все memory-срезы имеют data_state=unavailable либо нулевой валидный
  знаменатель, не делай вывод, что модели помнят, не помнят, знают или не знают
  бренд без веба. Упомяни ограничение ровно один раз — только отдельным
  элементом limitations и дословно: «Срез без веб-поиска не измерен: вывод о
  памяти моделей не формируется.» Не обсуждай memory, «знание без веба» и
  сравнение с веб-поиском в headline, verdict, executive_summary, sections или
  actions. Если недоступна лишь часть memory-срезов, ограничивай вывод только
  доступными семействами и не подменяй ими недоступные;
- если memory-срез имеет evidence_state=legacy_observational либо
  limitation_reason=legacy_memory_request_not_enforced, его агрегаты можно
  описывать только как «исторический срез, запрошенный без веб-поиска» с
  явным уточнением: ссылок и сигналов обращения к веб-инструментам не
  обнаружено, но техническое отключение веба в том запуске не аттестовано.
  Не называй это строгим знанием или памятью модели и не делай качественных
  выводов о содержании ответов сверх чисел из report_data. Добавь ровно один
  отдельный элемент limitations и дословно: «{CANONICAL_OBSERVATIONAL_MEMORY_LIMITATION}»;
- сравнение веб-поиска и памяти использует paired_web_lift с одинаковым
  составом систем и n_pairs>0; непарный Perplexity не включай в вывод о
  разрыве, а веб-срез никогда не подставляй вместо memory-среза. Если
  score_lift=null, но observed_difference задан из legacy-observational
  среза, разрешено только описательное сопоставление с указанным выше
  ограничением; не называй разницу эффектом веб-поиска;
- конкуренты названы только если они есть в нормализованной разметке;
- рекомендации связаны с конкретным наблюдением;
- ограничения говорят «экспресс-снимок», не раскрывают точный объём корпуса
  ответов моделей и не изображают статистическую репрезентативность; число
  проверенных страниц технического аудита, напротив, указывай явно;
- форма входа на содержательной странице не считается стеной авторизации,
если основной текст доступен;
- неизвестность не превращается в плохой результат;
- не делай причинных выводов там, где показана только связь.
- сам выбери количество содержательных разделов и приоритетных действий по
объёму и сложности доказательств; не отбрасывай отдельный значимый вывод ради
формального лимита и не дроби одну мысль без необходимости.
- пиши для руководителя и маркетолога, а не для разработчика пайплайна:
не показывай внутренние JSON-ключи, имена полей, enum-значения, artifact keys
или служебные идентификаторы. Переводи их в естественные русские формулировки:
«текст есть в серверном HTML», «уверенность средняя», «проверено две
содержательные страницы», «пять ответов содержат расхождения в фактах».
Сохраняй сами числа, знаменатели и смысл доказательства.

В selected_full_answers передана детерминированная выборка контекста из
корпуса, который прошёл независимого критика. Доступ к raw определяет
context_eligible/context_access, а не metric_eligible. Для
context_access=full_text связка «сценарий — raw-ответ» передана полностью и не
обрезана. Для context_access=metadata_only переданы только метаданные и
provenance: answer_text, annotation evidence и citations намеренно
отсутствуют. requested_mode показывает лишь запрошенный режим; verified_mode
показывает транспортно подтверждённый режим и для непроверенной строки равен
null. metric_eligible=true вместе с context_eligible=false допустимо только
для ограниченного legacy-observational агрегата: строка участвует в
программном числе, но не является качественным доказательством памяти модели.
answer_corpus_manifest описывает весь корпус, а
answer_selection_manifest доказывает покрытие выборки и связывает её с полным
корпусом. Доли и показатели не пересчитывай: числовой источник истины —
report_data. Строку с context_access=metadata_only разрешено использовать
только как контекст ограничения протокола; запрещено строить по её скрытому
содержанию вывод или причинное объяснение. Не показывай читателю внутренние
идентификаторы, hashes, версии контрактов или размер корпуса.

Если payload содержит semantic_review_to_fix и rejected_report, это ровно одна
ограниченная попытка исправить отклонённый текст. Исправь перечисленные
противоречия, сохрани все остальные подтверждённые выводы и не меняй числа.

Если payload содержит structure_validation_errors_to_fix и rejected_report,
это единственная отдельная попытка восстановить обязательную структуру
авторского кандидата. Заполни содержательные sections и actions, устрани
перечисленные структурные ошибки, сохрани подтверждённые выводы и числа. Не
добавляй semantic_review_to_fix и не выполняй самостоятельный новый анализ.

{LIVE_RUSSIAN_RULES}
""".strip()
    payload = _final_report_payload(public_report, answer_corpus)
    semantic_evidence_document = {
        "report_data": public_report,
        "selected_answer_context": payload["selected_full_answers"],
        "answer_selection_manifest": payload["answer_selection_manifest"],
    }
    preflight = _final_input_preflight(
        payload,
        reserve_tokens=FINAL_REPAIR_TOKEN_RESERVE,
    )
    preflight_error = None
    if not preflight["accepted"]:
        preflight_error = (
            "Final Opus input exceeds the configured context budget: "
            f"{preflight['estimated_input_tokens']} estimated tokens + "
            f"{preflight['repair_reserve_tokens']} repair reserve > "
            f"{preflight['input_token_budget']}."
        )
    await _save_artifact(
        run_id,
        stage_key="report",
        artifact_key="final_report_preflight",
        status="completed" if preflight["accepted"] else "failed",
        model=ANALYSIS_MODEL,
        input_json={
            "final_report_version": FINAL_REPORT_VERSION,
            "corpus_manifest_digest": answer_corpus["manifest"]["digest"],
            "selection_manifest_digest": (
                payload["answer_selection_manifest"]["digest"]
            ),
        },
        output_json=preflight,
        error_message=preflight_error,
        prompt_version=FINAL_REPORT_VERSION,
    )
    if preflight_error is not None:
        raise OpenRouterError(preflight_error)
    cached = await _artifact_output(
        run_id,
        "final_report",
        input_json=payload,
        model=ANALYSIS_MODEL,
        prompt_version=FINAL_REPORT_VERSION,
    )
    final_cache_candidate: dict[str, Any] | None = None
    if isinstance(cached, dict):
        cached = _sanitize_headline_emphasis(copy.deepcopy(cached))
        if not _validate_final_report(cached):
            final_cache_candidate = cached

    if final_cache_candidate is None:
        await _save_artifact(
            run_id,
            stage_key="report",
            artifact_key="final_report",
            status="running",
            model=ANALYSIS_MODEL,
            input_json=payload,
            prompt_version=FINAL_REPORT_VERSION,
        )
    try:
        candidate_raw_text: str | None = None
        candidate_usage: dict[str, Any] | None = None
        if final_cache_candidate is None:
            (
                candidate,
                candidate_raw_text,
                candidate_usage,
            ) = await _final_report_author_candidate(
                run_id,
                system=system,
                payload=payload,
            )
        else:
            candidate = final_cache_candidate

        semantic_attempt = 1
        semantic_review = await _final_report_semantic_review_artifact(
            run_id,
            public_report=public_report,
            candidate_report=candidate,
            selected_answer_context=payload["selected_full_answers"],
            answer_selection_manifest=payload["answer_selection_manifest"],
            attempt=semantic_attempt,
        )
        last_errors = report_semantic_blockers(
            candidate,
            public_report,
            semantic_review,
            evidence_document=semantic_evidence_document,
        )
        if not last_errors:
            if final_cache_candidate is not None:
                return candidate
            await _save_artifact(
                run_id,
                stage_key="report",
                artifact_key="final_report",
                status="completed",
                model=ANALYSIS_MODEL,
                input_json=payload,
                output_json=candidate,
                raw_text=candidate_raw_text,
                usage_json=candidate_usage,
                prompt_version=FINAL_REPORT_VERSION,
            )
            return candidate
        if semantic_review.get("verdict") == "block":
            await _invalidate_blocked_final_report_author_candidate(
                run_id,
                payload=payload,
                candidate=candidate,
                semantic_review=semantic_review,
                blockers=last_errors,
            )
            raise OpenRouterError(
                "Final report semantic gate blocked publication: "
                + "; ".join(last_errors)
            )

        rejected_report = candidate
        last_semantic_review = semantic_review
        for _attempt in range(MAX_FINAL_REPORT_REPAIRS):
            user_payload = dict(payload)
            user_payload["validation_errors_to_fix"] = last_errors
            user_payload["semantic_review_to_fix"] = last_semantic_review
            user_payload["rejected_report"] = rejected_report
            retry_preflight = _final_input_preflight(user_payload)
            if not retry_preflight["accepted"]:
                retry_error = (
                    "Final Opus retry input exceeds the configured context "
                    f"budget: {retry_preflight['estimated_input_tokens']} "
                    f"estimated tokens > "
                    f"{retry_preflight['input_token_budget']}."
                )
                await _save_artifact(
                    run_id,
                    stage_key="report",
                    artifact_key="final_report_preflight",
                    status="failed",
                    model=ANALYSIS_MODEL,
                    input_json={
                        "final_report_version": FINAL_REPORT_VERSION,
                        "corpus_manifest_digest": answer_corpus["manifest"][
                            "digest"
                        ],
                        "selection_manifest_digest": (
                            payload["answer_selection_manifest"]["digest"]
                        ),
                        "retry_after_validation": True,
                        "repair_kind": "semantic",
                    },
                    output_json=retry_preflight,
                    error_message=retry_error,
                    prompt_version=FINAL_REPORT_VERSION,
                )
                raise OpenRouterError(retry_error)
            result = await chat(
                model=ANALYSIS_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": json.dumps(user_payload, ensure_ascii=False),
                    },
                ],
                response_schema=FINAL_REPORT_SCHEMA,
                schema_name="aiv_final_report",
                reasoning_effort="high",
                max_tokens=30_000,
                temperature=0.15,
            )
            if not isinstance(result.parsed, dict):
                last_errors = ["Ответ не является объектом."]
                continue
            repaired = _sanitize_headline_emphasis(result.parsed)
            last_errors = _validate_final_report(repaired)
            if last_errors:
                continue
            semantic_attempt += 1
            semantic_review = await _final_report_semantic_review_artifact(
                run_id,
                public_report=public_report,
                candidate_report=repaired,
                selected_answer_context=payload["selected_full_answers"],
                answer_selection_manifest=payload["answer_selection_manifest"],
                attempt=semantic_attempt,
            )
            last_errors = report_semantic_blockers(
                repaired,
                public_report,
                semantic_review,
                evidence_document=semantic_evidence_document,
            )
            if last_errors:
                if semantic_review.get("verdict") == "block":
                    raise OpenRouterError(
                        "Final report semantic gate blocked publication: "
                        + "; ".join(last_errors)
                    )
                last_semantic_review = semantic_review
                rejected_report = repaired
                continue
            await _save_artifact(
                run_id,
                stage_key="report",
                artifact_key="final_report",
                status="completed",
                model=ANALYSIS_MODEL,
                input_json=payload,
                output_json=repaired,
                raw_text=result.text,
                usage_json=result.usage,
                prompt_version=FINAL_REPORT_VERSION,
            )
            return repaired
        raise OpenRouterError(
            "; ".join(last_errors) or "Final report validation failed"
        )
    except Exception as exc:
        await _save_artifact(
            run_id,
            stage_key="report",
            artifact_key="final_report",
            status="failed",
            model=ANALYSIS_MODEL,
            input_json=payload,
            error_message=str(exc),
            prompt_version=FINAL_REPORT_VERSION,
        )
        raise


def _eligible_illustration_answer_context(
    full_answers: list[dict[str, Any]] | None,
    *,
    limit: int = 8,
) -> tuple[list[dict[str, Any]], dict[str, int | bool]]:
    """Build visual context without exposing unattested panel content."""

    sanitized = [
        _final_model_answer_context_item(item)
        for item in (full_answers or [])
        if isinstance(item, dict)
    ]
    eligible = [
        item for item in sanitized if item.get("context_access") == "full_text"
    ]
    selected = []
    for item in eligible[:limit]:
        selected.append(
            {
                **item,
                "answer_text": str(item.get("answer_text") or "")[:3000],
            }
        )
    return selected, {
        "eligible_full_text_count": len(eligible),
        "selected_full_text_count": len(selected),
        "withheld_metadata_only_count": len(sanitized) - len(eligible),
        "unattested_raw_content_included": False,
    }


async def _illustration_concepts(
    run_id: str,
    public_report: dict[str, Any],
    evidence: list[dict[str, Any]],
    full_answers: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    system = f"""
Ты арт-директор сервиса AI Visibility команды RW+. Анализируемый сайт, его
бренд, продукт и рынок всегда бери только из переданного payload. Для сайта
rw.plus исследуемый бренд может совпадать с издателем сервиса — это нормально.
По уже рассчитанным результатам
подготовь ровно три содержательные концепции иллюстраций для итогового отчёта.
Это отдельный визуальный слой: не пиши аналитический отчёт и не меняй выводы.

Порядок и роли концепций:
1. technical_access — путь контента сайта к ИИ-краулерам: серверный HTML,
структурированные данные, клиентский рендеринг и реальные препятствия;
2. competitive_visibility — положение бренда в поле рекомендаций и заметные
конкуренты, только если они присутствуют в переданных данных;
3. web_memory_gap — различие между ответами с веб-поиском и знаниями из памяти.

Для каждой концепции:
- title, caption и alt_text напиши по-русски;
- report_data — единственный актуальный снимок расчётов. Каждый раз заново
  выводи из него title, caption, alt_text и core_claim; не воспроизводи текст
  прежней иллюстрации или числа, которых нет в этом снимке;
- если относящийся к выводу срез имеет data_state=unavailable, state=unknown,
  valid_answers=0 либо значение null, прямо скажи, что сопоставимых данных
  недостаточно. Не называй в таком случае долю, разрыв, количество конкретных
  ответов или направление преимущества;
- для сравнения веб-поиска и исторического среза используй только
  paired_web_lift. Если n_pairs=0 или observed_difference=null, не формулируй
  числовой вывод о разнице. Если observed_difference задан, это разница долей
  упоминаний в процентных пунктах на одинаковом наборе запросов, а не
  причинный эффект веб-поиска; при legacy-observational обязательно повтори
  ограничение аттестации режима;
- evidence_sample и selected_full_answers содержат только технически
  подтверждённые ответы. withheld_metadata_only_count в
  answer_context_contract — число намеренно скрытых строк, а не результат и
  не доказательство того, что модель что-либо знает или не знает;
- core_claim формулирует один проверяемый главный вывод без новых расчётов;
- evidence_paths содержит только существующие JSON Pointer пути из report_data.
  Для technical_access используй /technical/..., для competitive_visibility —
  /discovery/..., /competitors/... и /brand/..., для web_memory_gap —
  /discovery/..., /brand_knowledge/... и /methodology/.... Выбирай 2–6 путей;
- context_for_image подробно, по-английски, называет анализируемый бренд,
  объясняет его сайт, рынок, продукты, реальные рабочие среды, визуально
  узнаваемые для отрасли объекты и последствия вывода;
- creative_brief — это смелая авторская отправная точка, а не чертёж. Предложи
  запоминающуюся визуальную метафору, сцену, композицию, материалы, свет,
  эмоциональный тон и работу с целевой сущностью. Разрешены объём, изометрия,
  коллаж, архитектурное пространство, сюрреалистичный масштаб и драматичный
  кроп, если главный вывод остаётся понятным;
- каждая сцена должна быть узнаваема именно по теме этого клиента: используй
  его типичные продукты, инструменты, носители, рабочие процессы и контекст
  рынка. Если имя клиента заменить на компанию из другой отрасли и идея всё
  ещё подходит без изменений — концепция недостаточно конкретна;
- не используй склад, абстрактные колонны, архив, безымянный лабиринт,
  трубопровод или пустой футуристический интерфейс как универсальную метафору,
  если такие объекты не относятся к реальному бизнесу клиента;
- три идеи должны заметно отличаться метафорой, масштабом и пространственным
  приёмом. Не своди их к одинаковым блок-схемам;
- не добавляй бренды, препятствия, причинные связи и показатели, которых нет
в данных;
- не проси рисовать текст, числа, логотипы и легенды внутри картинки: название,
  точные метрики и объяснение покажет веб-страница.

{LIVE_RUSSIAN_RULES}
""".strip()
    selected_visual_answers, visual_context_contract = (
        _eligible_illustration_answer_context(full_answers)
    )
    payload = {
        "report_data": public_report,
        "evidence_sample": evidence,
        "selected_full_answers": selected_visual_answers,
        "answer_context_contract": visual_context_contract,
    }
    cached = await _artifact_output(
        run_id,
        "illustration_concepts",
        input_json=payload,
        model=ILLUSTRATION_CONCEPT_MODEL,
        prompt_version=ILLUSTRATION_CONCEPTS_VERSION,
    )
    if isinstance(cached, dict):
        cached = _normalize_unpaired_memory_illustration(cached, public_report)
        if not _validate_illustration_concepts(cached, public_report):
            return list(cached["illustrations"])

    last_errors: list[str] = []
    await _save_artifact(
        run_id,
        stage_key="report",
        artifact_key="illustration_concepts",
        status="running",
        model=ILLUSTRATION_CONCEPT_MODEL,
        input_json=payload,
        prompt_version=ILLUSTRATION_CONCEPTS_VERSION,
    )
    try:
        for _attempt in range(2):
            user_payload = dict(payload)
            if last_errors:
                user_payload["validation_errors_to_fix"] = last_errors
            result = await chat(
                model=ILLUSTRATION_CONCEPT_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": json.dumps(user_payload, ensure_ascii=False),
                    },
                ],
                response_schema=ILLUSTRATION_CONCEPTS_SCHEMA,
                schema_name="aiv_illustration_concepts",
                reasoning_effort="high",
                max_tokens=10_000,
                temperature=0.2,
            )
            if not isinstance(result.parsed, dict):
                last_errors = ["Ответ не является объектом."]
                continue
            result.parsed = _normalize_unpaired_memory_illustration(
                result.parsed,
                public_report,
            )
            last_errors = _validate_illustration_concepts(
                result.parsed,
                public_report,
            )
            if last_errors:
                continue
            await _save_artifact(
                run_id,
                stage_key="report",
                artifact_key="illustration_concepts",
                status="completed",
                model=ILLUSTRATION_CONCEPT_MODEL,
                input_json=payload,
                output_json=result.parsed,
                raw_text=result.text,
                usage_json=result.usage,
                prompt_version=ILLUSTRATION_CONCEPTS_VERSION,
            )
            return list(result.parsed["illustrations"])
        raise OpenRouterError(
            "; ".join(last_errors) or "Illustration concept validation failed"
        )
    except Exception as exc:
        await _save_artifact(
            run_id,
            stage_key="report",
            artifact_key="illustration_concepts",
            status="failed",
            model=ILLUSTRATION_CONCEPT_MODEL,
            input_json=payload,
            error_message=str(exc),
            prompt_version=ILLUSTRATION_CONCEPTS_VERSION,
        )
        raise


def _illustration_generation_concept(
    concept: dict[str, Any],
    *,
    sequence: int,
    fact_context: dict[str, Any],
) -> dict[str, Any]:
    return {
        **concept,
        "sequence": sequence,
        "fact_contract": fact_context,
    }


def _illustration_prompt(
    concept: dict[str, Any],
    *,
    brand_name: str,
    sequence: int,
) -> str:
    role_goals = {
        1: (
            "Show how the site's meaningful content reaches AI systems and "
            "where the evidence confirms access, an obstacle, or an unresolved state."
        ),
        2: (
            "Show the evaluated brand and its commercially relevant portfolio "
            "inside the real field of choices and alternatives. Never assume absence."
        ),
        3: (
            "Show the measured relationship between current web-grounded answers "
            "and model memory: stronger, weaker, similar, or insufficient evidence."
        ),
    }
    safe_concept = json.loads(json.dumps(concept, ensure_ascii=False))
    if brand_name.strip():
        brand_pattern = re.compile(
            rf"(?<![\w]){re.escape(brand_name.strip())}(?![\w])",
            re.IGNORECASE,
        )

        def replace_brand(value: Any) -> Any:
            if isinstance(value, str):
                return brand_pattern.sub("the analyzed client", value)
            if isinstance(value, list):
                return [replace_brand(item) for item in value]
            if isinstance(value, dict):
                return {key: replace_brand(item) for key, item in value.items()}
            return value

        safe_concept = replace_brand(safe_concept)
    fact_contract = safe_concept.get("fact_contract") or {}
    brand_context = fact_contract.get("brand") or {}
    creative_brief = safe_concept.get("creative_brief") or {}
    client_context = {
        "name": "the analyzed client",
        "site_type": brand_context.get("site_type"),
        "category": brand_context.get("category"),
        "products": brand_context.get("products") or [],
        "positioning": brand_context.get("positioning"),
    }
    return f"""
Create one bold, authored editorial infographic for a premium AI-visibility
report. It must feel specific to this website, its product category and this
finding — not like a generic flowchart, dashboard or technology decoration.

ANALYZED CLIENT
{json.dumps(client_context, ensure_ascii=False, indent=2)}

ROLE GOAL
{role_goals.get(sequence)}

IMMUTABLE EVIDENCE
{json.dumps(fact_contract, ensure_ascii=False, indent=2)}

CORE CLAIM
{safe_concept.get("core_claim")}

EDITORIAL CONTEXT
{safe_concept.get("context_for_image")}

ART-DIRECTOR'S STARTING POINT
{json.dumps(creative_brief, ensure_ascii=False, indent=2)}

The evidence determines meaning. The art direction is intentionally open.
Invent the visual world, composition, perspective, scale, spatial rhythm,
materials, lighting and metaphor. You may use dimensional forms, architectural
space, tactile collage, editorial data art, dramatic cropping or a surreal but
legible scene. Treat the creative brief as a starting point, not a blueprint.

Build the image around one dominant and memorable visual idea. Make the core
claim immediately understandable and let every expressive choice reinforce it.
Use objects, environments, media, tools and workflows native to the analyzed
client's actual business. The client should be recognizable by subject matter
even with no logo and no caption. A generic warehouse, archive, colonnade,
labyrinth, pipeline or anonymous sci-fi interface is unacceptable unless it is
literally part of the client's market.

Semantic guardrails:
- do not introduce a brand, competitor, number, ranking, barrier, cause or
  relationship that is absent from IMMUTABLE EVIDENCE;
- keep unknown states visibly unresolved instead of explaining their cause;
- decorative details may create energy and atmosphere, but must not resemble
  additional data points, rankings or causal arrows;
- exact figures remain in the surrounding HTML charts and tables unless the
  evidence explicitly marks an image encoding as exact;
- do not invent or approximate a logo. Avoid watermarks, long text and
  pseudo-text; the webpage supplies the title, labels and caption. Small
  accurate category-native interface details are allowed when they help make
  the client's field recognizable.

Use a cold blue-gray canvas (#CDD5DE) and near-black foundation as house-style
anchors. Use #FF324B sparingly for the evaluated target or the single decisive
accent. The palette does not require flatness: depth, texture, light and
material contrast are welcome. 16:9, 2K.
""".strip()


def _illustration_cache_matches(
    illustration: ReportIllustration,
    prompt: str,
) -> bool:
    return bool(
        illustration.file_url
        and illustration.generation_prompt == prompt
        and illustration.model == settings.OPENROUTER_IMAGE_MODEL
        and not illustration.error_message
        and (illustration.usage_json or {}).get("generation_version")
        == ILLUSTRATION_GENERATION_VERSION
    )


def _illustration_fact_context(
    public_report: dict[str, Any],
    sequence: int,
    evidence_paths: list[str] | None = None,
) -> dict[str, Any]:
    context: dict[str, Any] = {
        "brand": public_report.get("brand") or {},
        "key_metrics": public_report.get("key_metrics") or {},
    }
    if sequence == 1:
        context["technical"] = public_report.get("technical") or {}
    elif sequence == 2:
        context["discovery"] = public_report.get("discovery") or {}
        context["competitors"] = public_report.get("competitors") or []
    else:
        context["discovery"] = public_report.get("discovery") or {}
        context["brand_knowledge"] = public_report.get("brand_knowledge") or {}
        context["methodology"] = public_report.get("methodology") or {}
    resolved: dict[str, Any] = {}
    for path in evidence_paths or []:
        try:
            resolved[path] = _resolve_json_pointer(public_report, path)
        except KeyError:
            continue
    if resolved:
        context["resolved_evidence"] = resolved
    return context


def _illustration_review_errors(review: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if review.get("usable") is not True:
        errors.append("Проверяющая модель не считает изображение пригодным.")
    if review.get("facts_grounded") is not True:
        errors.append("Главный визуальный вывод противоречит данным.")
    if review.get("claim_readable") is not True:
        errors.append("Главный вывод не считывается из композиции.")
    scores = review.get("scores")
    if not isinstance(scores, dict):
        errors.append("Проверка качества не вернула поле scores.")
        context_specificity = None
    else:
        context_specificity = scores.get("context_specificity")
    if (
        not isinstance(context_specificity, int)
        or isinstance(context_specificity, bool)
    ):
        errors.append(
            "Проверка качества не оценила контекстную специфичность."
        )
    elif context_specificity < 4:
        errors.append(
            "Изображение недостаточно узнаваемо по рынку, продуктам и "
            "рабочему контексту анализируемого клиента."
        )
    for key in ("unsupported_assertions", "hard_blockers"):
        values = review.get(key)
        if not isinstance(values, list):
            errors.append(f"Проверка качества не вернула поле {key}.")
            continue
        errors.extend(
            str(value).strip()
            for value in values
            if isinstance(value, str) and value.strip()
        )
    if not isinstance(review.get("visible_text_problems"), list):
        errors.append(
            "Проверка качества не вернула поле visible_text_problems."
        )
    return list(dict.fromkeys(errors))


async def _review_illustration(
    run_id: str,
    *,
    sequence: int,
    concept: dict[str, Any],
    fact_context: dict[str, Any],
    generation_prompt: str,
    image_content: bytes,
    media_type: str,
) -> dict[str, Any]:
    image_sha256 = hashlib.sha256(image_content).hexdigest()
    generation_prompt_sha256 = hashlib.sha256(
        generation_prompt.encode("utf-8")
    ).hexdigest()
    payload = {
        "sequence": sequence,
        "concept": concept,
        "fact_context": fact_context,
        "generation_prompt_sha256": generation_prompt_sha256,
        "image_sha256": image_sha256,
        "quality_requirements": {
            "core_claim_readable": True,
            "facts_grounded": True,
            "client_context_specific": True,
            "no_unsupported_assertions": True,
            "no_logos_or_prominent_readable_text": True,
            "minor_unreadable_textures_are_advisory": True,
            "creative_stylization_allowed": True,
        },
    }
    artifact_key = f"illustration_qa_{sequence}_{image_sha256[:16]}"
    cached = await _artifact_output(
        run_id,
        artifact_key,
        input_json=payload,
        model=PROCESSING_MODEL,
        prompt_version=ILLUSTRATION_QA_VERSION,
    )
    if isinstance(cached, dict):
        return cached

    system = """
Ты визуальный редактор премиального аналитического отчёта. Сначала независимо
опиши inferred_message — что зритель действительно считывает с изображения.
Затем сравни это прочтение, само изображение, core_claim и fact_context.

Контекстная конкретность обязательна. Изображение должно узнаваться как сцена
из рынка и продуктового мира анализируемого клиента: по его носителям,
инструментам, рабочим процессам и коммерческим объектам. Красивую метафору,
которую без изменений можно поставить в отчёт компании из другой отрасли,
считай непригодной: usable=false и context_specificity не выше 3.

Жёсткие блокеры:
- изображение сообщает противоположный главный вывод;
- оно добавляет неподтверждённый бренд, конкурента, число, рейтинг, причину,
  преимущество или причинную связь;
- неизвестность превращена в выдуманное объяснение;
- видны логотип, водяной знак, длинная читаемая надпись либо текст, который
  меняет главный смысл;
- декор полностью скрывает смысл.

Объём, перспектива, свет, фактура, коллаж, необычная метафора, мягкая тень,
градиент и стилистическая неровность сами по себе не являются ошибками.
Мелкие нечитаемые штрихи или текстоподобная фактура — замечание в
visible_text_problems, но не hard_blocker и не причина ставить usable=false,
если факты верны и центральная мысль ясна. Псевдотекст блокирует публикацию
только когда выглядит как значимая подпись, интерфейс, бренд или утверждение.
Точные числа проверяй только тогда, когда концепция явно требует буквального
кодирования числа внутри изображения. Иначе числа остаются в HTML отчёта.

usable=true, если нет жёстких блокеров и основной смысл не противоречит данным.
claim_readable показывает, можно ли понять центральную мысль без подписи.
Низкий эстетический балл не делает facts_grounded=false: оцени его отдельно
в scores по шкале 1–5. hard_blockers, unsupported_assertions и
visible_text_problems содержат только конкретные видимые проблемы.
retry_instruction напиши по-английски как позитивное предложение другой
метафоры, масштаба, ракурса и материалов, сохранив факт-контракт.
""".strip()
    await _save_artifact(
        run_id,
        stage_key="report",
        artifact_key=artifact_key,
        status="running",
        model=PROCESSING_MODEL,
        input_json=payload,
        prompt_version=ILLUSTRATION_QA_VERSION,
    )
    data_url = (
        f"data:{media_type};base64,"
        f"{base64.b64encode(image_content).decode('ascii')}"
    )
    try:
        result = await chat(
            model=PROCESSING_MODEL,
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(payload, ensure_ascii=False),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                    ],
                },
            ],
            response_schema=ILLUSTRATION_QA_SCHEMA,
            schema_name=f"aiv_illustration_qa_{sequence}",
            reasoning_effort="high",
            max_tokens=4500,
            temperature=0.05,
        )
        if not isinstance(result.parsed, dict):
            raise OpenRouterError("Image quality review is not an object")
        await _save_artifact(
            run_id,
            stage_key="report",
            artifact_key=artifact_key,
            status="completed",
            model=PROCESSING_MODEL,
            input_json=payload,
            output_json=result.parsed,
            raw_text=result.text,
            usage_json=result.usage,
            prompt_version=ILLUSTRATION_QA_VERSION,
        )
        return result.parsed
    except Exception as exc:
        await _save_artifact(
            run_id,
            stage_key="report",
            artifact_key=artifact_key,
            status="failed",
            model=PROCESSING_MODEL,
            input_json=payload,
            error_message=str(exc),
            prompt_version=ILLUSTRATION_QA_VERSION,
        )
        raise


def _illustration_retry_prompt(
    base_prompt: str,
    *,
    errors: list[str],
    retry_instruction: str,
    attempt: int,
) -> str:
    if attempt <= 1 and not errors and not retry_instruction.strip():
        return base_prompt
    if not errors:
        return f"""
{base_prompt}

Create an independent second art direction. Preserve the immutable evidence and
core claim, but change the central metaphor, scale, camera angle, spatial rhythm
and material language. Aim for a bolder, more memorable editorial composition
than the first candidate. Do not imitate a generic diagram.
""".strip()
    return f"""
{base_prompt}

QUALITY REVIEW REJECTED THE PREVIOUS CANDIDATE.
Generation attempt: {attempt}.

Positive reconstruction plan from the visual reviewer:
{retry_instruction.strip() or "Create a clearer fact-grounded editorial infographic."}

Discard the previous composition. Invent a different metaphor, scale, camera
angle, spatial rhythm, materials and lighting while preserving the immutable
evidence and the core claim. Keep all semantic guardrails from the base prompt.
""".strip()


async def _generate_reviewed_image(
    run_id: str,
    *,
    sequence: int,
    concept: dict[str, Any],
    fact_context: dict[str, Any],
    base_prompt: str,
    initial_review: dict[str, Any] | None = None,
) -> tuple[ImageResult, dict[str, Any], str, int]:
    feedback_errors = (
        _illustration_review_errors(initial_review)
        if isinstance(initial_review, dict)
        else []
    )
    retry_instruction = (
        str(initial_review.get("retry_instruction") or "")
        if isinstance(initial_review, dict)
        else ""
    )
    last_errors = list(feedback_errors)
    candidates: list[
        tuple[ImageResult, dict[str, Any], str, list[str]]
    ] = []
    history: list[dict[str, Any]] = []
    attempts_made = 0

    for attempt in range(1, ILLUSTRATION_MAX_ATTEMPTS + 1):
        attempts_made = attempt
        candidate_prompt = _illustration_retry_prompt(
            base_prompt,
            errors=feedback_errors,
            retry_instruction=retry_instruction,
            attempt=attempt,
        )
        try:
            image = await generate_image(
                prompt=candidate_prompt,
                aspect_ratio="16:9",
                resolution="2K",
            )
            if image.media_type == "image/svg+xml":
                raise OpenRouterError(
                    "SVG is not accepted because it cannot pass raster visual QA"
                )
            review = await _review_illustration(
                run_id,
                sequence=sequence,
                concept=concept,
                fact_context=fact_context,
                generation_prompt=candidate_prompt,
                image_content=image.content,
                media_type=image.media_type,
            )
            last_errors = _illustration_review_errors(review)
            candidates.append((image, review, candidate_prompt, last_errors))
            history.append(
                {
                    "candidate": attempt,
                    "image_sha256": hashlib.sha256(image.content).hexdigest(),
                    "usable": not last_errors,
                    "errors": last_errors,
                    "scores": review.get("scores") or {},
                    "generation_usage": image.usage,
                }
            )
            if last_errors:
                feedback_errors = last_errors
                retry_instruction = str(review.get("retry_instruction") or "")
            else:
                feedback_errors = []
                retry_instruction = ""
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            last_errors = [f"Техническая ошибка кандидата: {str(exc)[:300]}"]
            feedback_errors = last_errors
            retry_instruction = (
                "Create a fresh, legible editorial infographic with one clear "
                "visual thesis grounded in the immutable evidence."
            )
            history.append(
                {
                    "candidate": attempt,
                    "usable": False,
                    "errors": last_errors,
                    "scores": {},
                }
            )
        if (
            attempt >= ILLUSTRATION_QUALITY_ATTEMPTS
            and any(not candidate[3] for candidate in candidates)
        ):
            break
        if attempt < ILLUSTRATION_MAX_ATTEMPTS:
            logger.info(
                "Illustration %s for run %s completed candidate %s",
                sequence,
                run_id,
                attempt,
            )

    usable = [candidate for candidate in candidates if not candidate[3]]
    if usable:
        def candidate_score(
            candidate: tuple[ImageResult, dict[str, Any], str, list[str]],
        ) -> int:
            scores = candidate[1].get("scores") or {}
            return sum(
                int(scores.get(key) or 0)
                for key in (
                    "context_specificity",
                    "visual_story",
                    "distinctiveness",
                    "hierarchy",
                    "craft",
                    "richness",
                )
            )

        image, review, accepted_prompt, _errors = max(
            usable,
            key=candidate_score,
        )
        selected_review = dict(review)
        selected_review["_candidate_history"] = history
        selected_review["_selection_score"] = candidate_score(
            (image, review, accepted_prompt, [])
        )
        return (
            image,
            selected_review,
            accepted_prompt,
            attempts_made,
        )

    raise OpenRouterError(
        "Illustration quality gate rejected all candidates: "
        + "; ".join(last_errors[:6])
    )


async def _generate_illustrations(
    run_id: str,
    *,
    brand_name: str,
    concepts: list[dict[str, Any]],
    public_report: dict[str, Any],
) -> list[dict[str, Any]]:
    if len(concepts) != 3:
        raise OpenRouterError(
            "The final report must contain exactly three illustration concepts"
        )

    safe_run_id = re.sub(r"[^a-zA-Z0-9_-]", "", run_id)
    target_dir = GENERATED_DIR / safe_run_id
    target_dir.mkdir(parents=True, exist_ok=True)
    generation_semaphore = asyncio.Semaphore(ILLUSTRATION_ROLE_CONCURRENCY)
    report_write_lock = asyncio.Lock()
    completed_roles = 0

    await update_progress(
        run_id,
        stage="report",
        percent=91,
        detail="Создаём и проверяем схемы: 0 из 3.",
        eta_seconds=450,
    )

    async def mark_role_completed() -> None:
        nonlocal completed_roles
        async with report_write_lock:
            completed_roles += 1
            await update_progress(
                run_id,
                stage="report",
                percent=91 + completed_roles * 2,
                detail=f"Создаём и проверяем схемы: {completed_roles} из 3.",
                eta_seconds=max(90, (3 - completed_roles) * 150),
            )

    async def generate_role(
        sequence: int,
        concept: dict[str, Any],
    ) -> dict[str, Any]:
        async with generation_semaphore:
            fact_context = _illustration_fact_context(
                public_report,
                sequence,
                [
                    str(path)
                    for path in concept.get("evidence_paths") or []
                    if isinstance(path, str)
                ],
            )
            generation_concept = _illustration_generation_concept(
                concept,
                sequence=sequence,
                fact_context=fact_context,
            )
            prompt = _illustration_prompt(
                generation_concept,
                brand_name=brand_name,
                sequence=sequence,
            )
            title = str(concept.get("title") or f"Схема {sequence}")[:300]
            caption = str(concept.get("caption") or "")
            alt_text = str(concept.get("alt_text") or "")

            async with SessionLocal() as session:
                illustration = (
                    await session.execute(
                        select(ReportIllustration).where(
                            ReportIllustration.run_id == run_id,
                            ReportIllustration.sequence == sequence,
                        )
                    )
                ).scalar_one_or_none()

            initial_review: dict[str, Any] | None = None
            if illustration and _illustration_cache_matches(illustration, prompt):
                filename = Path(str(illustration.file_url)).name
                cached_path = target_dir / filename
                if filename and cached_path.is_file():
                    cached_content = cached_path.read_bytes()
                    media_type = {
                        ".jpg": "image/jpeg",
                        ".jpeg": "image/jpeg",
                        ".webp": "image/webp",
                    }.get(cached_path.suffix.lower(), "image/png")
                    stored_usage = dict(illustration.usage_json or {})
                    accepted_prompt = str(
                        stored_usage.get("accepted_generation_prompt") or prompt
                    )
                    try:
                        initial_review = await _review_illustration(
                            run_id,
                            sequence=sequence,
                            concept=generation_concept,
                            fact_context=fact_context,
                            generation_prompt=accepted_prompt,
                            image_content=cached_content,
                            media_type=media_type,
                        )
                        if not _illustration_review_errors(initial_review):
                            stored_usage.update(
                                {
                                    "quality_review": initial_review,
                                    "quality_model": PROCESSING_MODEL,
                                    "quality_version": ILLUSTRATION_QA_VERSION,
                                    "generation_version": (
                                        ILLUSTRATION_GENERATION_VERSION
                                    ),
                                    "image_sha256": hashlib.sha256(
                                        cached_content
                                    ).hexdigest(),
                                }
                            )
                            async with report_write_lock:
                                async with SessionLocal() as session:
                                    current = (
                                        await session.execute(
                                            select(ReportIllustration).where(
                                                ReportIllustration.run_id == run_id,
                                                ReportIllustration.sequence == sequence,
                                            )
                                        )
                                    ).scalar_one()
                                    current.title = title
                                    current.caption = caption
                                    current.alt_text = alt_text
                                    current.usage_json = stored_usage
                                    current.error_message = None
                                    await session.commit()
                            result = {
                                "sequence": sequence,
                                "title": title,
                                "caption": caption,
                                "alt_text": alt_text,
                                "file_url": illustration.file_url,
                            }
                            await mark_role_completed()
                            return result
                    except Exception:
                        logger.exception(
                            "Cached illustration %s could not pass visual QA for run %s",
                            sequence,
                            run_id,
                        )

            async with report_write_lock:
                async with SessionLocal() as session:
                    illustration = (
                        await session.execute(
                            select(ReportIllustration).where(
                                ReportIllustration.run_id == run_id,
                                ReportIllustration.sequence == sequence,
                            )
                        )
                    ).scalar_one_or_none()
                    if illustration is None:
                        illustration = ReportIllustration(
                            run_id=run_id,
                            sequence=sequence,
                            title=title,
                            caption=caption,
                            alt_text=alt_text,
                            generation_prompt=prompt,
                            model=settings.OPENROUTER_IMAGE_MODEL,
                        )
                        session.add(illustration)
                    else:
                        illustration.title = title
                        illustration.caption = caption
                        illustration.alt_text = alt_text
                        illustration.generation_prompt = prompt
                        illustration.model = settings.OPENROUTER_IMAGE_MODEL
                        illustration.file_url = None
                        illustration.usage_json = None
                        illustration.error_message = None
                    await session.commit()

            try:
                image, review, accepted_prompt, quality_attempts = (
                    await _generate_reviewed_image(
                        run_id,
                        sequence=sequence,
                        concept=generation_concept,
                        fact_context=fact_context,
                        base_prompt=prompt,
                        initial_review=initial_review,
                    )
                )
                filename = f"{sequence:02d}.{image.extension}"
                target = target_dir / filename
                temporary = target.with_suffix(target.suffix + ".tmp")
                temporary.write_bytes(image.content)
                temporary.replace(target)
                file_url = f"/static/generated/{safe_run_id}/{filename}"
                async with report_write_lock:
                    async with SessionLocal() as session:
                        illustration = (
                            await session.execute(
                                select(ReportIllustration).where(
                                    ReportIllustration.run_id == run_id,
                                    ReportIllustration.sequence == sequence,
                                )
                            )
                        ).scalar_one()
                        illustration.file_url = file_url
                        illustration.generation_prompt = prompt
                        illustration.usage_json = {
                            "generation": image.usage,
                            "generation_version": ILLUSTRATION_GENERATION_VERSION,
                            "quality_review": review,
                            "quality_model": PROCESSING_MODEL,
                            "quality_version": ILLUSTRATION_QA_VERSION,
                            "quality_attempts": quality_attempts,
                            "accepted_generation_prompt": accepted_prompt,
                            "image_sha256": hashlib.sha256(image.content).hexdigest(),
                        }
                        illustration.error_message = None
                        await session.commit()
                result = {
                    "sequence": sequence,
                    "title": title,
                    "caption": caption,
                    "alt_text": alt_text,
                    "file_url": file_url,
                }
            except Exception as exc:
                logger.warning(
                    "Illustration %s failed for run %s: %s",
                    sequence,
                    run_id,
                    type(exc).__name__,
                )
                async with report_write_lock:
                    async with SessionLocal() as session:
                        illustration = (
                            await session.execute(
                                select(ReportIllustration).where(
                                    ReportIllustration.run_id == run_id,
                                    ReportIllustration.sequence == sequence,
                                )
                            )
                        ).scalar_one()
                        illustration.error_message = str(exc)[:1000]
                        await session.commit()
                result = {
                    "sequence": sequence,
                    "title": title,
                    "caption": caption,
                    "alt_text": alt_text,
                    "file_url": None,
                }

            await mark_role_completed()
            return result

    return list(
        await asyncio.gather(
            *(
                generate_role(sequence, concept)
                for sequence, concept in enumerate(concepts[:3], start=1)
            )
        )
    )


def _reuse_saved_illustration_assets(
    saved_illustrations: list[dict[str, Any]],
    refreshed_concepts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pair immutable saved image assets with freshly derived report copy."""

    saved_by_sequence: dict[int, dict[str, Any]] = {}
    for position, item in enumerate(saved_illustrations, start=1):
        if not isinstance(item, dict):
            continue
        sequence = _non_negative_int(item.get("sequence"))
        if sequence is None or sequence < 1:
            sequence = position
        saved_by_sequence.setdefault(sequence, item)

    reused: list[dict[str, Any]] = []
    for sequence, concept in enumerate(refreshed_concepts, start=1):
        saved = saved_by_sequence.get(sequence)
        if saved is None or not isinstance(concept, dict):
            continue
        reused.append(
            {
                "sequence": sequence,
                "title": str(concept.get("title") or f"Схема {sequence}")[:300],
                "caption": str(concept.get("caption") or ""),
                "alt_text": (
                    str(saved.get("alt_text"))
                    if isinstance(saved.get("alt_text"), str)
                    and str(saved.get("alt_text")).strip()
                    else str(concept.get("alt_text") or "")
                ),
                "file_url": copy.deepcopy(saved.get("file_url")),
            }
        )
    return reused


async def _synchronize_reused_illustration_metadata(
    session: Any,
    *,
    run_id: str,
    illustrations: list[dict[str, Any]],
) -> None:
    """Align database copy without changing saved image generation state."""

    for item in illustrations:
        sequence = _non_negative_int(item.get("sequence"))
        if sequence is None or sequence < 1:
            raise OpenRouterError("Reused illustration has no valid sequence")
        updated = await session.execute(
            update(ReportIllustration)
            .where(
                ReportIllustration.run_id == run_id,
                ReportIllustration.sequence == sequence,
            )
            .values(
                title=str(item.get("title") or f"Схема {sequence}")[:300],
                caption=str(item.get("caption") or ""),
                alt_text=str(item.get("alt_text") or ""),
            )
        )
        if updated.rowcount != 1:
            raise OpenRouterError(
                f"Saved illustration row is missing for sequence {sequence}"
            )


def _fallback_reused_illustration_concepts() -> list[dict[str, Any]]:
    """Return number-free, universally valid copy for immutable saved assets."""

    return [
        {
            "role": "technical_access",
            "title": "Как сайт открывается ИИ-краулерам",
            "caption": (
                "Схема связывает серверный HTML, правила доступа и "
                "сущностную разметку на проверенных страницах."
            ),
            "alt_text": (
                "Схема технических условий, от которых зависит чтение сайта "
                "ИИ-краулерами."
            ),
        },
        {
            "role": "competitive_visibility",
            "title": "Где бренд появляется среди альтернатив",
            "caption": (
                "Схема показывает место бренда среди названных альтернатив "
                "по подтверждённым ответам моделей."
            ),
            "alt_text": (
                "Схема присутствия бренда и альтернатив в ответах моделей."
            ),
        },
        {
            "role": "web_memory_gap",
            "title": "Сопоставимого среза для сравнения пока нет",
            "caption": (
                "Подтверждённых пар ответов с веб-поиском и без него "
                "недостаточно, поэтому направление разницы не оценивается."
            ),
            "alt_text": (
                "Схема отмечает отсутствие сопоставимого подтверждённого "
                "среза для сравнения режимов."
            ),
        },
    ]


async def _run_reused_report_branches(
    run_id: str,
    *,
    public_report: dict[str, Any],
    evidence: list[dict[str, Any]],
    answer_corpus: dict[str, Any],
    saved_illustrations: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Refresh analysis and illustration copy without generating image files."""

    if not saved_illustrations:
        return await _final_report(run_id, public_report, answer_corpus), []

    async def build_refreshed_concepts() -> list[dict[str, Any]]:
        try:
            return await _illustration_concepts(
                run_id,
                public_report,
                evidence,
                answer_corpus["answers"],
            )
        except asyncio.CancelledError:
            raise
        except OpenRouterError as exc:
            logger.warning(
                "Illustration copy validation failed for saved run %s; "
                "using deterministic fallback (%s)",
                run_id,
                type(exc).__name__,
            )
            fallback = _fallback_reused_illustration_concepts()
            await _save_artifact(
                run_id,
                stage_key="report",
                artifact_key="illustration_concepts_fallback",
                status="completed",
                model=None,
                input_json={
                    "reason": "saved_asset_copy_validation_failed",
                    "source_version": ILLUSTRATION_CONCEPTS_VERSION,
                },
                output_json={"illustrations": fallback},
                prompt_version=ILLUSTRATION_COPY_FALLBACK_VERSION,
            )
            return fallback

    async with asyncio.TaskGroup() as task_group:
        final_task = task_group.create_task(
            _final_report(run_id, public_report, answer_corpus),
            name=f"aiv-report-reanalysis-{run_id}",
        )
        concepts_task = task_group.create_task(
            build_refreshed_concepts(),
            name=f"aiv-report-visual-copy-{run_id}",
        )

    return (
        final_task.result(),
        _reuse_saved_illustration_assets(
            saved_illustrations,
            concepts_task.result(),
        ),
    )


async def _run_report_branches(
    run_id: str,
    *,
    public_report: dict[str, Any],
    evidence: list[dict[str, Any]],
    answer_corpus: dict[str, Any],
    brand_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    async def build_visuals() -> list[dict[str, Any]]:
        try:
            concepts = await _illustration_concepts(
                run_id,
                public_report,
                evidence,
                answer_corpus["answers"],
            )
            await update_progress(
                run_id,
                stage="report",
                percent=89,
                detail="Визуальные концепции готовы. Создаём три инфографики.",
                eta_seconds=300,
            )
            illustrations = await _generate_illustrations(
                run_id,
                brand_name=brand_name,
                concepts=concepts,
                public_report=public_report,
            )
            await update_progress(
                run_id,
                stage="report",
                percent=97,
                detail="Визуальный слой готов. Завершаем аналитику.",
                eta_seconds=120,
            )
            return illustrations
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "Illustration branch failed for run %s; publishing analysis",
                run_id,
            )
            return []

    async with asyncio.TaskGroup() as task_group:
        final_task = task_group.create_task(
            _final_report(
                run_id,
                public_report,
                answer_corpus,
            ),
            name=f"aiv-report-analysis-{run_id}",
        )
        illustrations_task = task_group.create_task(
            build_visuals(),
            name=f"aiv-report-visuals-{run_id}",
        )

    return final_task.result(), illustrations_task.result()


async def _review_technical_summary(
    run_id: str,
    technical: dict[str, Any],
) -> dict[str, Any]:
    return await _structured_artifact(
        run_id,
        stage_key="technical_access",
        artifact_key="technical_review",
        schema=TECHNICAL_REVIEW_SCHEMA,
        schema_name="aiv_technical_review",
        system=f"""
Проверь технический аудит сайта как старший специалист по доступности
контента для ИИ-краулеров. Работай только с переданными фактами.

Обязательные правила:
- форма входа не является стеной авторизации, если на странице есть
  содержательный основной текст;
- отсутствие данных — unknown, а не блокировка;
- CSR оценивай по исходному HTML, отдельно от SSR и статической выдачи;
- различай robots.txt, сетевую блокировку, WAF и пустую клиентскую оболочку;
- строки pages с is_utility=true — служебные подтверждения или технические
  адреса. Они показаны для прозрачности, но исключены из score и не должны
  превращаться в вывод о доступности содержательной части сайта;
- не меняй программно рассчитанный score;
- score — это индекс технической готовности по нескольким измеренным
  сигналам, а не процент доступности и не доказательство индексации;
- crawler user-agent пробы подтверждают только HTTP-ответ и читаемый HTML
  для имитированного user-agent. Не утверждай, что реальные системы уже
  обошли, проиндексировали или «прочитали» сайт;
- если у страницы body_truncated=true или structured_data_complete=false,
  отсутствие текста, разметки или признаков рендеринга на ней считается
  unknown. Не превращай усечённый ответ в отрицательный вывод;
- различай любую вспомогательную Schema.org-разметку и сущностные типы
  Organization, Product, Service, Article. Не пиши «Schema.org отсутствует»,
  если в structured_data_types присутствует хотя бы один тип;
- robots.state=partial означает частичные ограничения. Даже когда выбранные
  адреса разрешены, не называй robots.txt полностью открытым;
- связывай score только с coverage.evaluated_pages: пиши «на проверенных
  страницах», не распространяй результат на весь сайт;
- если coverage.coverage_state равно limited или unknown, назови ограничение
  в overall_conclusion и limitations; неизвестный полный размер сайта не
  означает полный охват;
- вывод связывай с тем, что сможет прочитать система и как это влияет на
  обнаружение бренда.

{LIVE_RUSSIAN_RULES}
""".strip(),
        user_payload=technical,
        max_tokens=9000,
        prompt_version=TECHNICAL_REVIEW_VERSION,
    )


async def _prepare_analysis_foundation(
    run_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build independent technical and semantic foundations in parallel."""

    technical, site_context = await asyncio.gather(
        _technical_summary(run_id),
        _site_context(run_id),
    )
    async with asyncio.TaskGroup() as task_group:
        technical_review_task = task_group.create_task(
            _review_technical_summary(run_id, technical),
            name=f"aiv-technical-review-{run_id}",
        )
        profile_task = task_group.create_task(
            _classify_site(run_id, site_context),
            name=f"aiv-site-profile-{run_id}",
        )
    return (
        technical,
        technical_review_task.result(),
        profile_task.result(),
        site_context,
    )


async def _uses_canonical_intent_taxonomy(run_id: str) -> bool:
    """Keep saved legacy answers labeled by the taxonomy that produced them."""

    async with SessionLocal() as session:
        version = (
            await session.execute(
                select(RunArtifact.prompt_version).where(
                    RunArtifact.run_id == run_id,
                    RunArtifact.artifact_key == "prompt_set",
                )
            )
        ).scalar_one_or_none()
    return version == PROMPT_SET_VERSION


async def _finish_saved_answer_analysis(
    run_id: str,
    *,
    profile: dict[str, Any],
    technical: dict[str, Any],
    technical_review: dict[str, Any],
    regenerate_illustrations: bool = True,
) -> None:
    catalog_answers = await _answers_for_catalog(run_id)
    if not catalog_answers:
        raise OpenRouterError("No saved model answers are available for reanalysis")
    catalog = await _entity_catalog(run_id, profile, catalog_answers)
    await _annotate_answers(run_id, profile, catalog)
    rows = await _metric_rows(
        run_id,
        annotation_input_sha256=_annotation_context_sha256(profile, catalog),
    )
    metrics = _compute_metrics(rows, profile, catalog)
    expected_corpus_cells = await _expected_corpus_cells(run_id, rows)
    catalog, rows, metrics, critic_gate = await _run_analysis_critic_loop(
        run_id,
        profile=profile,
        catalog=catalog,
        rows=rows,
        metrics=metrics,
        expected_corpus_cells=expected_corpus_cells,
    )
    await _save_artifact(
        run_id,
        stage_key="knowledge_gap",
        artifact_key="metrics",
        status="completed",
        model=None,
        prompt_version=METRICS_VERSION,
        input_json={
            "annotation_digest": _stable_json_sha256(
                [
                    {
                        "answer_id": row["answer_id"],
                        "annotation": row["annotation"],
                    }
                    for row in rows
                ]
            ),
            "critic_gate_digest": _stable_json_sha256(critic_gate),
        },
        output_json=metrics,
    )
    public_report = _build_public_report(
        profile=profile,
        technical=technical,
        technical_review=technical_review,
        metrics=metrics,
        canonical_intent_taxonomy=await _uses_canonical_intent_taxonomy(run_id),
    )
    await update_progress(
        run_id,
        stage="report",
        percent=82,
        detail=(
            "Параллельно собираем аналитику и три визуальные концепции."
            if regenerate_illustrations
            else "Пересобираем расчёты и интерпретацию из сохранённых ответов."
        ),
        eta_seconds=360,
    )
    evidence = await _evidence_sample(run_id)
    answer_corpus = await _full_answer_context(
        run_id,
        critic_gate=critic_gate,
        critic_rows=rows,
        expected_corpus_cells=expected_corpus_cells,
    )
    if regenerate_illustrations:
        final, illustrations = await _run_report_branches(
            run_id,
            public_report=public_report,
            evidence=evidence,
            answer_corpus=answer_corpus,
            brand_name=str(profile.get("brand_name") or "бренд"),
        )
    else:
        async with SessionLocal() as session:
            saved_report = (
                await session.execute(
                    select(Run.report_json).where(Run.id == run_id)
                )
            ).scalar_one_or_none()
        saved_illustrations = (
            copy.deepcopy(saved_report.get("illustrations") or [])
            if isinstance(saved_report, dict)
            else []
        )
        final, illustrations = await _run_reused_report_branches(
            run_id,
            public_report=public_report,
            evidence=evidence,
            answer_corpus=answer_corpus,
            saved_illustrations=saved_illustrations,
        )
    markdown = _render_markdown(final)
    site_preview = await get_saved_site_preview(run_id)
    report_json = {
        **public_report,
        "narrative": {
            "headline": final.get("headline"),
            "headline_emphasis": final.get("headline_emphasis") or [],
            "verdict": final.get("verdict"),
            "executive_summary": final.get("executive_summary"),
            "actions": final.get("actions") or [],
        },
        "illustrations": illustrations,
        **({"site_preview": site_preview} if site_preview else {}),
    }
    owner = lease_owner_for(run_id)
    report_conditions = [Run.id == run_id]
    if owner is not None:
        report_conditions.extend(
            (
                Run.execution_slot == 1,
                Run.lease_owner == owner,
                Run.status.in_(
                    (
                        RunStatus.pending,
                        RunStatus.crawling,
                        RunStatus.analyzing,
                    )
                ),
            )
        )
    async with SessionLocal() as session:
        saved = await session.execute(
            update(Run)
            .where(*report_conditions)
            .values(
                analysis_markdown=markdown,
                report_json=report_json,
            )
            .returning(Run.id)
        )
        saved_run_id = saved.scalar_one_or_none()
        if saved_run_id is None:
            await session.rollback()
        else:
            if not regenerate_illustrations:
                await _synchronize_reused_illustration_metadata(
                    session,
                    run_id=run_id,
                    illustrations=illustrations,
                )
            await session.commit()
    if saved_run_id is None:
        if owner is not None:
            raise RunLeaseLostError(f"Run lease lost for {run_id}")
        raise LookupError(f"Run not found: {run_id}")
    completed = await complete_run(run_id)
    if not completed:
        if owner is not None:
            raise RunLeaseLostError(
                f"Run lease lost before terminal completion for {run_id}"
            )
        raise LookupError(f"Run could not be completed: {run_id}")


async def reprocess_saved_answers(run_id: str) -> None:
    """Rebuild annotations, metrics and report without calling the model panel."""

    try:
        await update_progress(
            run_id,
            stage="knowledge_gap",
            percent=70,
            detail="Переанализируем сохранённые ответы без повторного опроса моделей.",
            eta_seconds=720,
            status=RunStatus.analyzing,
        )
        technical, technical_review, profile, _site_context_value = (
            await _prepare_analysis_foundation(run_id)
        )
        await _finish_saved_answer_analysis(
            run_id,
            profile=profile,
            technical=technical,
            technical_review=technical_review,
            regenerate_illustrations=False,
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("Saved-answer reanalysis failed for run %s", run_id)
        await fail_run(
            run_id,
            "Не удалось завершить переанализ сохранённых ответов. "
            "Исходные ответы остались в базе.",
        )


async def analyze_run(run_id: str) -> None:
    try:
        await update_progress(
            run_id,
            stage="scenario_design",
            percent=28,
            detail="Определяем бренд, продукты и реальные задачи выбора.",
            eta_seconds=1080,
            status=RunStatus.analyzing,
        )
        await apply_ua_conditional_block(run_id)
        technical, technical_review, profile, site_context = (
            await _prepare_analysis_foundation(run_id)
        )
        await update_progress(
            run_id,
            stage="scenario_design",
            percent=31,
            detail="Проверяем рынок, аудитории и критерии выбора по источникам.",
            eta_seconds=960,
        )
        market_research = await _market_research(
            run_id,
            profile,
            site_context,
        )
        await update_progress(
            run_id,
            stage="scenario_design",
            percent=34,
            detail="Строим сценарии из подтверждённых задач выбора.",
            eta_seconds=840,
        )
        prompt_set = await _generate_prompt_set(
            run_id,
            profile,
            site_context.get("requested_site"),
            market_research=market_research,
        )
        prompts = await _persist_prompts(run_id, prompt_set)
        await update_progress(
            run_id,
            stage="web_visibility",
            percent=38,
            detail="Запрашиваем независимые ответы с актуальным веб-поиском.",
            eta_seconds=900,
        )
        await _run_panel(
            run_id,
            prompts,
            mode="web",
            start_percent=38,
            end_percent=64,
        )
        await update_progress(
            run_id,
            stage="knowledge_gap",
            percent=65,
            detail="Проверяем знания моделей без веб-поиска.",
            eta_seconds=600,
        )
        await _run_panel(
            run_id,
            prompts,
            mode="memory",
            start_percent=65,
            end_percent=72,
        )
        await _finish_saved_answer_analysis(
            run_id,
            profile=profile,
            technical=technical,
            technical_review=technical_review,
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("AIV analysis failed for run %s", run_id)
        await fail_run(
            run_id,
            "Проверка прервалась во время анализа. Нажмите «Продолжить»: "
            "сервис использует уже сохранённые ответы и завершит недостающие этапы.",
        )
