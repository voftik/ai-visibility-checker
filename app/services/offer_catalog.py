from __future__ import annotations

"""Deterministic, source-bound offer catalog contracts.

This module is deliberately independent from the crawler, database, and LLM
transport.  It gives the pipeline a code-owned boundary between probabilistic
site/profile extraction and later market research / INTENT generation:

* every accepted product, service, or business direction is bound to an exact
  excerpt in a content-addressed source unit and to an explicit grammatical
  actor/possessor relationship with the client;
* generic market vocabulary is not promoted to a client offer merely because
  it occurs in a query or answer;
* duplicate and overflow decisions are explicit and deterministic;
* all downstream artifacts are bound by SHA-256 digests, so a resumed run
  cannot mix a rebuilt profile/catalog with stale prompts or answers;
* long research payloads are split into lossless UTF-8 shards.  The shard size
  is a processing target, never a content limit, and there is no shard-count or
  document-length ceiling.

The maximum of ten accepted offers is a product-analysis scope contract, not a
limit on evidence length or LLM response length.
"""

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
import re
import unicodedata
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

import tldextract


OFFER_CATALOG_VERSION = "aiv-offer-catalog-v5"
OFFER_EVIDENCE_VERSION = "aiv-offer-evidence-v2"
OFFER_USER_JOB_EVIDENCE_VERSION = "aiv-offer-user-job-evidence-v1"
OFFER_CANDIDATE_NORMALIZATION_VERSION = "aiv-offer-candidate-normalization-v2"
DOMAIN_RESEARCH_PAYLOAD_VERSION = "aiv-domain-research-payload-v1"
DOMAIN_RESEARCH_MANIFEST_VERSION = "aiv-domain-research-manifest-v1"
OFFER_CLUSTER_VERSION = "aiv-offer-cluster-v1"
OFFER_CATALOG_ADMISSION_VERSION = "aiv-offer-catalog-admission-v2"
CLIENT_IDENTITY_ADMISSION_VERSION = "aiv-client-identity-admission-v2"
PROMPT_FOUNDATION_VERSION = "aiv-prompt-foundation-v1"
ANSWER_SET_RECEIPT_VERSION = "aiv-answer-set-receipt-v1"
UPSTREAM_DIGESTS_VERSION = "aiv-upstream-artifact-digests-v1"
MAX_ACCEPTED_OFFERS = 10
INTENT_CODES = ("I", "E", "T", "NB", "NAV", "TR")

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SPACE_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[^\w]+", re.UNICODE)

# These are category vocabulary, not proprietary entities.  The list is
# intentionally conservative: a phrase is generic only when the complete
# normalized name matches.  "Realweb DOOH" therefore remains a distinct name,
# while bare "DOOH" needs an explicit commercial binding to the client.
_GENERIC_OFFER_TERMS = frozenset(
    {
        "aeo",
        "ai search",
        "ai visibility",
        "brandformance",
        "contextual advertising",
        "content marketing",
        "digital marketing",
        "dooh",
        "geo",
        "influencer marketing",
        "media buying",
        "performance marketing",
        "programmatic",
        "programmatic advertising",
        "search engine optimization",
        "seo",
        "social media marketing",
        "smm",
        "web development",
        "контекстная реклама",
        "контент маркетинг",
        "маркетинг",
        "медиабаинг",
        "медийная реклама",
        "перформанс маркетинг",
        "программатик",
        "разработка сайтов",
        "таргетированная реклама",
    }
)

_OWNERSHIP_CLAUSE_SPLIT_RE = re.compile(
    r"(?:[\r\n.!?;:]+|,\s+(?=(?:а|но|однако|but|while|whereas)\b))",
    re.IGNORECASE,
)
_OWNERSHIP_ACTION_RE = re.compile(
    r"(?<!\S)(?:"
    r"offer(?:s|ed|ing)?|provid(?:e|es|ed|ing)|deliver(?:s|ed|ing)?|"
    r"develop(?:s|ed|ing)?|build(?:s|ing)?|launch(?:es|ed|ing)?|"
    r"creat(?:e|es|ed|ing)|operat(?:e|es|ed|ing)|"
    r"speciali[sz](?:e|es|ed|ing)(?:\s+in)?|"
    r"nudi|nudimo|nudite|nude|"
    r"pruža|pružamo|pružate|pružaju|"
    r"stvara|stvaramo|stvarate|stvaraju|"
    r"предлага(?:ю|ем|ет|ют|л|ла|ли)|"
    r"оказыва(?:ю|ем|ет|ют|л|ла|ли)|"
    r"предоставля(?:ю|ем|ет|ют|л|ла|ли)|"
    r"разрабатыва(?:ю|ем|ет|ют|л|ла|ли)|"
    r"развива(?:ю|ем|ет|ют|л|ла|ли)|"
    r"запуска(?:ю|ем|ет|ют|л|ла|ли)|"
    r"созда(?:ю|ем|ет|ют|л|ла|ли)|"
    r"выпуска(?:ю|ем|ет|ют|л|ла|ли)|"
    r"специализиру(?:юсь|емся|ется|ются|лся|лась|лись)"
    r"(?:\s+на)?"
    r")(?!\S)",
    re.IGNORECASE,
)
_OWNERSHIP_SAFE_MODIFIERS = frozenset(
    {
        "also",
        "currently",
        "directly",
        "do",
        "now",
        "nowadays",
        "активно",
        "также",
        "теперь",
        "уже",
        "сейчас",
    }
)
_OWNERSHIP_OBJECT_PREFIX_TOKEN_RE = re.compile(
    r"(?:a|an|the|its|our|own|new|including|"
    r"наш(?:а|е|и|у|ей|его|ему|им|их)?|"
    r"собственн\w*|нов\w*|включая|по|"
    r"product\w*|service\w*|platform\w*|solution\w*|offering\w*|"
    r"practice\w*|tool\w*|"
    r"продукт\w*|услуг\w*|сервис\w*|платформ\w*|решени\w*|"
    r"направлени\w*|практик\w*|инструмент\w*)",
    re.IGNORECASE,
)
_FOREIGN_ATTRIBUTION_MARKERS = frozenset(
    {"by", "from", "of", "от", "у"}
)
_OWNERSHIP_TYPE_PATTERN = (
    r"(?:product|service|platform|solution|offering|practice|tool|"
    r"продукт\w*|услуг\w*|сервис\w*|платформ\w*|решени\w*|"
    r"направлени\w*|практик\w*|инструмент\w*)"
)
_OWNERSHIP_CONNECTOR_PATTERN = (
    r"(?:a|an|the|is|of|from|by|это|компани\w*|агентств\w*|"
    r"бренд\w*|групп\w*|от|для)"
)
_FIRST_PERSON_ACTORS = ("we", "мы")
_FIRST_PERSON_POSSESSIVES = (
    "our",
    "наш",
    "наша",
    "наше",
    "наши",
)
_QUOTE_MARKERS = frozenset({'"', "'", "«", "»", "“", "”", "„", "‟"})

# First-person possessives may be the grammatical actor only together with a
# narrow organization-role noun.  We deliberately do not accept an arbitrary
# ``our <words>`` span: a first-party page can discuss "our partner agency" or
# quote a client, and that must not transfer the partner's offers to the site.
_FIRST_PERSON_ORGANIZATION_ACTOR_RE = re.compile(
    r"(?<!\S)(?:"
    r"our\s+(?:studio|agency|company|team|shop|salon|clinic|practice|business)|"
    r"наш(?:а|е|и)?\s+(?:студия|агентство|компания|команда|салон|клиника|практика|бизнес)|"
    r"naš(?:a|e|i)?\s+(?:studio|agencija|tvrtka|kompanija|tim|salon|klinika|praksa)"
    r")(?!\S)",
    re.IGNORECASE,
)

# A narrow relative-clause bridge used by ordinary studio copy such as
# ``our studio offers a professional environment where experienced artists
# create custom tattoos`` and its Croatian equivalent.  This is a grammatical
# shape, not a character-distance window.
_FIRST_PARTY_STAFF_BRIDGE_RE = re.compile(
    r"(?:"
    r"(?:a\s+)?(?:professional\s+)?environment\s+(?:where|in\s+which)\s+"
    r"(?:experienced|skilled|our)\s+(?:artists?|specialists?|professionals?)|"
    r"profesionalno\s+okruženje\s+u\s+kojem\s+"
    r"(?:iskusni|vješti|naši)\s+(?:umjetnici|stručnjaci)"
    r")",
    re.IGNORECASE,
)
_FIRST_PARTY_SCOPE_ACTION_RE = re.compile(
    r"(?<!\S)(?:offer(?:s|ed|ing)?|provid(?:e|es|ed|ing)|"
    r"nudi|nudimo|nudite|nude|predlaže|predlažemo)(?!\S)",
    re.IGNORECASE,
)
_COORDINATED_OFFER_CONJUNCTIONS = frozenset(
    {"and", "or", "i", "ili", "te", "и", "или"}
)


class OfferCatalogError(ValueError):
    """Base class for a rejected catalog or downstream contract."""


class OfferEvidenceError(OfferCatalogError):
    """Raised when source identity or literal evidence cannot be proven."""


class PromptCoverageError(OfferCatalogError):
    """Raised when six INTENT prompts do not account for every offer cluster."""


class OfferCatalogAdmissionError(PromptCoverageError):
    """Raised before generic INTENT prompts can mask a missing offer catalog."""


class ResumeCompatibilityError(OfferCatalogError):
    """Raised when current and persisted analysis foundations differ."""


class OfferKind(str, Enum):
    PRODUCT = "product"
    SERVICE = "service"
    DIRECTION = "direction"


class DispositionDecision(str, Enum):
    ACCEPTED = "accepted"
    DUPLICATE = "duplicate"
    REJECTED = "rejected"
    OVERFLOW = "overflow"


class OfferCatalogAdmissionStatus(str, Enum):
    OFFERS_PRESENT = "offers_present"
    ZERO_OFFERS_ADMITTED = "zero_offers_admitted"
    ZERO_OFFERS_BLOCKED = "zero_offers_blocked"


class ZeroOfferSiteKind(str, Enum):
    NOT_APPLICABLE = "not_applicable"
    NONCOMMERCIAL = "noncommercial"
    NO_PUBLIC_OFFER = "no_public_offer"
    COMMERCIAL_OR_UNKNOWN = "commercial_or_unknown"


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise OfferCatalogError(
            f"Value is not canonically JSON-serializable: {exc}"
        ) from exc


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def artifact_digest(value: Any) -> str:
    """Return a deterministic digest for one JSON-compatible artifact."""

    if hasattr(value, "as_dict"):
        value = value.as_dict()
    return _sha256_text(_canonical_json(value))


def _require_sha256(value: str, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise OfferCatalogError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _normalize_phrase(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold().replace("ё", "е")
    normalized = _WORD_RE.sub(" ", normalized)
    return _SPACE_RE.sub(" ", normalized).strip()


def _normalized_unique(values: Iterable[str]) -> tuple[str, ...]:
    by_normalized: dict[str, str] = {}
    for raw in values:
        if not isinstance(raw, str):
            raise OfferCatalogError("Aliases and user jobs must be strings")
        value = _SPACE_RE.sub(" ", raw).strip()
        if not value:
            continue
        key = _normalize_phrase(value)
        if key and key not in by_normalized:
            by_normalized[key] = value
    return tuple(by_normalized[key] for key in sorted(by_normalized))


# An empty catalog is meaningful only for an explicitly classified site that
# does not expose public commercial offers.  These phrases are deliberately
# narrow.  Broad labels such as "portal", "corporate site" or
# "informational site" are not enough: commercial sites use them too.
_NONCOMMERCIAL_SITE_KIND_MARKERS = tuple(
    _normalize_phrase(value)
    for value in (
        "некоммерческий сайт",
        "некоммерческий информационный ресурс",
        "некоммерческая организация",
        "некоммерческий проект",
        "noncommercial website",
        "non commercial website",
        "nonprofit website",
        "nonprofit organization",
        "not for profit",
    )
)
_NO_PUBLIC_OFFER_SITE_KIND_MARKERS = tuple(
    _normalize_phrase(value)
    for value in (
        "без публичного предложения",
        "без публичных предложений",
        "публичного предложения нет",
        "публичных предложений нет",
        "публичное предложение отсутствует",
        "публичные предложения отсутствуют",
        "не предлагает продукты или услуги",
        "не предлагает продукты и услуги",
        "нет публичных продуктов или услуг",
        "no public offer",
        "no public offering",
        "no public products or services",
        "does not offer products or services",
    )
)

# An unrelated exact quote must not license an empty commercial catalog.  The
# full saved source corpus is therefore scanned for conservative, high-signal
# commercial contradictions.  Clauses that explicitly say that no public
# offer exists are excluded before this scan, otherwise the words "offer" or
# "products" inside a negation would become their own contradiction.
_COMMERCIAL_SOURCE_SIGNAL_MARKERS = tuple(
    _normalize_phrase(value)
    for value in (
        "купить",
        "заказать",
        "оставить заявку",
        "записаться",
        "получить консультацию",
        "наши услуги",
        "наши продукты",
        "оказываем услуги",
        "предлагаем услуги",
        "предлагаем продукты",
        "цена",
        "цены",
        "стоимость",
        "тариф",
        "тарифы",
        "оплата",
        "подписка",
        "магазин",
        "корзина",
        "buy",
        "order",
        "book now",
        "get a quote",
        "contact sales",
        "our services",
        "our products",
        "we offer",
        "pricing",
        "price",
        "plans",
        "subscription",
        "shop",
        "cart",
        "checkout",
    )
)
_SOURCE_CLAUSE_SPLIT_RE = re.compile(r"[\r\n.!?;]+")


def _contains_site_kind_marker(value: str, markers: Sequence[str]) -> bool:
    normalized = _normalize_phrase(str(value or ""))
    return bool(normalized) and any(marker in normalized for marker in markers)


def _matching_site_kind_marker(
    value: str,
    markers: Sequence[str],
) -> str | None:
    normalized = _normalize_phrase(str(value or ""))
    if not normalized:
        return None
    return next((marker for marker in markers if marker in normalized), None)


def _profile_model_evidence(profile: Mapping[str, Any]) -> tuple[str, ...]:
    raw = profile.get("evidence")
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        return ()
    return _normalized_unique(
        value for value in raw if isinstance(value, str) and value.strip()
    )


def _normalize_evidence_whitespace(value: str) -> str:
    return _SPACE_RE.sub(" ", str(value or "")).strip()


def _source_bound_profile_evidence_receipts(
    *,
    evidence: Sequence[str],
    source_units: Iterable[SourceUnit | Mapping[str, Any]],
) -> tuple[tuple[dict[str, Any], ...], str]:
    source_map: dict[str, SourceUnit] = {}
    for raw_source in source_units:
        source = (
            raw_source
            if isinstance(raw_source, SourceUnit)
            else SourceUnit.from_mapping(raw_source)
        )
        existing = source_map.get(source.source_unit_id)
        if existing is not None and existing != source:
            raise OfferCatalogAdmissionError(
                f"Conflicting admission source units use ID {source.source_unit_id!r}"
            )
        source_map[source.source_unit_id] = source

    source_manifest_digest = artifact_digest(
        [source_map[key].descriptor() for key in sorted(source_map)]
    )
    receipts: list[dict[str, Any]] = []
    for quote in evidence:
        normalized_quote = _normalize_evidence_whitespace(quote)
        if not normalized_quote:
            continue
        for source_unit_id in sorted(source_map):
            source = source_map[source_unit_id]
            normalized_source = _normalize_evidence_whitespace(source.text)
            if normalized_quote not in normalized_source:
                continue
            receipts.append(
                {
                    "source_unit_id": source.source_unit_id,
                    "source_url": source.source_url,
                    "source_sha256": source.source_sha256,
                    "evidence_excerpt": normalized_quote,
                    "evidence_sha256": _sha256_text(normalized_quote),
                    "evidence_utf8_length": len(normalized_quote.encode("utf-8")),
                    "match_policy": "exact_after_whitespace_normalization",
                }
            )
    receipts.sort(
        key=lambda item: (
            str(item["source_unit_id"]),
            str(item["evidence_sha256"]),
        )
    )
    return tuple(receipts), source_manifest_digest


def _site_kind_evidence_receipts(
    receipts: Sequence[Mapping[str, Any]],
    *,
    site_kind: ZeroOfferSiteKind,
) -> tuple[dict[str, Any], ...]:
    if site_kind is ZeroOfferSiteKind.NONCOMMERCIAL:
        markers = _NONCOMMERCIAL_SITE_KIND_MARKERS
    elif site_kind is ZeroOfferSiteKind.NO_PUBLIC_OFFER:
        markers = _NO_PUBLIC_OFFER_SITE_KIND_MARKERS
    else:
        return ()

    proven: list[dict[str, Any]] = []
    for raw_receipt in receipts:
        receipt = json.loads(_canonical_json(raw_receipt))
        marker = _matching_site_kind_marker(
            str(receipt.get("evidence_excerpt") or ""),
            markers,
        )
        if marker is None:
            continue
        receipt["site_kind_proof"] = site_kind.value
        receipt["site_kind_marker"] = marker
        proven.append(receipt)
    return tuple(proven)


def _source_commercial_signal_receipts(
    source_units: Iterable[SourceUnit | Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    receipts: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for raw_source in source_units:
        source = (
            raw_source
            if isinstance(raw_source, SourceUnit)
            else SourceUnit.from_mapping(raw_source)
        )
        for raw_clause in _SOURCE_CLAUSE_SPLIT_RE.split(source.text):
            clause = _normalize_evidence_whitespace(raw_clause)
            normalized_clause = _normalize_phrase(clause)
            if not normalized_clause:
                continue
            if _contains_site_kind_marker(
                normalized_clause,
                _NO_PUBLIC_OFFER_SITE_KIND_MARKERS,
            ):
                continue
            for marker in _COMMERCIAL_SOURCE_SIGNAL_MARKERS:
                if not _contains_normalized_phrase(normalized_clause, marker):
                    continue
                key = (source.source_unit_id, marker, _sha256_text(clause))
                if key in seen:
                    continue
                seen.add(key)
                receipts.append(
                    {
                        "source_unit_id": source.source_unit_id,
                        "source_url": source.source_url,
                        "source_sha256": source.source_sha256,
                        "signal_marker": marker,
                        "signal_excerpt": clause,
                        "signal_sha256": _sha256_text(clause),
                        "match_policy": "exact_clause_after_whitespace_normalization",
                    }
                )
    receipts.sort(
        key=lambda item: (
            str(item["source_unit_id"]),
            str(item["signal_marker"]),
            str(item["signal_sha256"]),
        )
    )
    return tuple(receipts)


def _profile_has_commercial_offer_signals(profile: Mapping[str, Any]) -> bool:
    products = profile.get("products")
    if (
        isinstance(products, Sequence)
        and not isinstance(products, (str, bytes))
        and any(str(value or "").strip() for value in products)
    ):
        return True

    candidates = profile.get("offer_candidates")
    if isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes)):
        if any(
            isinstance(value, Mapping)
            and value.get("commercially_relevant") is True
            for value in candidates
        ):
            return True

    entities = profile.get("entity_scope")
    commercial_entity_types = {"business_unit", "product", "service", "platform"}
    if isinstance(entities, Sequence) and not isinstance(entities, (str, bytes)):
        if any(
            isinstance(value, Mapping)
            and value.get("commercially_relevant") is True
            and str(value.get("entity_type") or "") in commercial_entity_types
            for value in entities
        ):
            return True
    return False


@dataclass(frozen=True, slots=True)
class OfferCatalogAdmission:
    """Code-owned decision for allowing analysis to continue past the catalog.

    The site-kind phrase and confidence come from the model-built profile.  A
    conservative code allowlist, source-profile evidence receipt and explicit
    contradiction checks decide whether an empty catalog is admissible.
    """

    status: OfferCatalogAdmissionStatus
    site_kind: ZeroOfferSiteKind
    catalog_digest: str
    profile_digest: str
    accepted_offer_count: int
    site_type: str
    business_model: str
    model_confidence: str
    model_evidence_count: int
    model_evidence_digest: str
    source_manifest_digest: str
    source_evidence_receipts: tuple[dict[str, Any], ...]
    source_evidence_receipt_digest: str
    source_commercial_signal_receipts: tuple[dict[str, Any], ...]
    source_commercial_signal_receipt_digest: str
    source_absence_claims_allowed: bool
    crawl_coverage_state: str
    reason_codes: tuple[str, ...]
    admission_digest: str
    version: str = OFFER_CATALOG_ADMISSION_VERSION

    @property
    def allowed(self) -> bool:
        return self.status is not OfferCatalogAdmissionStatus.ZERO_OFFERS_BLOCKED

    def _body(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "status": self.status.value,
            "site_kind": self.site_kind.value,
            "catalog_digest": self.catalog_digest,
            "profile_digest": self.profile_digest,
            "accepted_offer_count": self.accepted_offer_count,
            "site_type": self.site_type,
            "business_model": self.business_model,
            "model_confidence": self.model_confidence,
            "model_evidence_count": self.model_evidence_count,
            "model_evidence_digest": self.model_evidence_digest,
            "source_manifest_digest": self.source_manifest_digest,
            "source_evidence_receipts": [
                json.loads(_canonical_json(item))
                for item in self.source_evidence_receipts
            ],
            "source_evidence_receipt_digest": (
                self.source_evidence_receipt_digest
            ),
            "source_commercial_signal_receipts": [
                json.loads(_canonical_json(item))
                for item in self.source_commercial_signal_receipts
            ],
            "source_commercial_signal_receipt_digest": (
                self.source_commercial_signal_receipt_digest
            ),
            "source_absence_claims_allowed": self.source_absence_claims_allowed,
            "crawl_coverage_state": self.crawl_coverage_state,
            "reason_codes": list(self.reason_codes),
        }

    def as_dict(self) -> dict[str, Any]:
        value = self._body()
        value["allowed"] = self.allowed
        value["admission_digest"] = self.admission_digest
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "OfferCatalogAdmission":
        version = str(value.get("version") or "")
        if version != OFFER_CATALOG_ADMISSION_VERSION:
            raise OfferCatalogAdmissionError(
                "Unsupported offer catalog admission version"
            )
        raw_reasons = value.get("reason_codes")
        raw_receipts = value.get("source_evidence_receipts")
        raw_signal_receipts = value.get("source_commercial_signal_receipts")
        if isinstance(raw_reasons, (str, bytes)) or not isinstance(
            raw_reasons, Sequence
        ):
            raise OfferCatalogAdmissionError(
                "Offer catalog admission reason_codes are invalid"
            )
        if (
            isinstance(raw_receipts, (str, bytes))
            or not isinstance(raw_receipts, Sequence)
            or any(not isinstance(item, Mapping) for item in raw_receipts)
        ):
            raise OfferCatalogAdmissionError(
                "Offer catalog admission source receipts are invalid"
            )
        if (
            isinstance(raw_signal_receipts, (str, bytes))
            or not isinstance(raw_signal_receipts, Sequence)
            or any(not isinstance(item, Mapping) for item in raw_signal_receipts)
        ):
            raise OfferCatalogAdmissionError(
                "Offer catalog admission commercial signal receipts are invalid"
            )
        try:
            admission = cls(
                status=OfferCatalogAdmissionStatus(str(value.get("status") or "")),
                site_kind=ZeroOfferSiteKind(str(value.get("site_kind") or "")),
                catalog_digest=str(value.get("catalog_digest") or ""),
                profile_digest=str(value.get("profile_digest") or ""),
                accepted_offer_count=int(value.get("accepted_offer_count")),
                site_type=str(value.get("site_type") or ""),
                business_model=str(value.get("business_model") or ""),
                model_confidence=str(value.get("model_confidence") or ""),
                model_evidence_count=int(value.get("model_evidence_count")),
                model_evidence_digest=str(value.get("model_evidence_digest") or ""),
                source_manifest_digest=str(
                    value.get("source_manifest_digest") or ""
                ),
                source_evidence_receipts=tuple(
                    json.loads(_canonical_json(item)) for item in raw_receipts
                ),
                source_evidence_receipt_digest=str(
                    value.get("source_evidence_receipt_digest") or ""
                ),
                source_commercial_signal_receipts=tuple(
                    json.loads(_canonical_json(item))
                    for item in raw_signal_receipts
                ),
                source_commercial_signal_receipt_digest=str(
                    value.get("source_commercial_signal_receipt_digest") or ""
                ),
                source_absence_claims_allowed=(
                    value.get("source_absence_claims_allowed") is True
                ),
                crawl_coverage_state=str(
                    value.get("crawl_coverage_state") or ""
                ),
                reason_codes=tuple(str(item) for item in raw_reasons),
                admission_digest=str(value.get("admission_digest") or ""),
                version=version,
            )
        except (TypeError, ValueError) as exc:
            raise OfferCatalogAdmissionError(
                "Offer catalog admission fields are invalid"
            ) from exc
        admission.validate()
        if value.get("allowed") is not admission.allowed:
            raise OfferCatalogAdmissionError(
                "Offer catalog admission allowed flag is inconsistent"
            )
        return admission

    def validate(self) -> None:
        _require_sha256(self.catalog_digest, field="catalog_digest")
        _require_sha256(self.profile_digest, field="profile_digest")
        _require_sha256(self.model_evidence_digest, field="model_evidence_digest")
        _require_sha256(self.source_manifest_digest, field="source_manifest_digest")
        _require_sha256(
            self.source_evidence_receipt_digest,
            field="source_evidence_receipt_digest",
        )
        _require_sha256(
            self.source_commercial_signal_receipt_digest,
            field="source_commercial_signal_receipt_digest",
        )
        _require_sha256(self.admission_digest, field="admission_digest")
        if self.version != OFFER_CATALOG_ADMISSION_VERSION:
            raise OfferCatalogAdmissionError(
                "Unsupported offer catalog admission version"
            )
        if (
            self.accepted_offer_count < 0
            or self.accepted_offer_count > MAX_ACCEPTED_OFFERS
        ):
            raise OfferCatalogAdmissionError(
                "Offer catalog admission count is outside product scope"
            )
        if self.model_evidence_count < 0:
            raise OfferCatalogAdmissionError(
                "Offer catalog admission evidence count is invalid"
            )
        if type(self.source_absence_claims_allowed) is not bool:
            raise OfferCatalogAdmissionError(
                "Offer catalog absence-claim flag is invalid"
            )
        if not self.crawl_coverage_state:
            raise OfferCatalogAdmissionError(
                "Offer catalog crawl coverage state is missing"
            )
        if artifact_digest(
            list(self.source_evidence_receipts)
        ) != self.source_evidence_receipt_digest:
            raise OfferCatalogAdmissionError(
                "Offer catalog admission source receipt digest mismatch"
            )
        for receipt in self.source_evidence_receipts:
            if (
                not str(receipt.get("source_unit_id") or "")
                or not str(receipt.get("source_url") or "")
                or receipt.get("match_policy")
                != "exact_after_whitespace_normalization"
            ):
                raise OfferCatalogAdmissionError(
                    "Offer catalog admission source receipt is incomplete"
                )
            _require_sha256(
                str(receipt.get("source_sha256") or ""),
                field="source_receipt.source_sha256",
            )
            _require_sha256(
                str(receipt.get("evidence_sha256") or ""),
                field="source_receipt.evidence_sha256",
            )
            evidence_excerpt = str(receipt.get("evidence_excerpt") or "")
            if (
                not evidence_excerpt
                or _sha256_text(evidence_excerpt)
                != str(receipt.get("evidence_sha256") or "")
            ):
                raise OfferCatalogAdmissionError(
                    "Offer catalog admission source receipt excerpt is invalid"
                )
        if artifact_digest(
            list(self.source_commercial_signal_receipts)
        ) != self.source_commercial_signal_receipt_digest:
            raise OfferCatalogAdmissionError(
                "Offer catalog admission commercial signal receipt digest mismatch"
            )
        for receipt in self.source_commercial_signal_receipts:
            if (
                not str(receipt.get("source_unit_id") or "")
                or not str(receipt.get("source_url") or "")
                or not str(receipt.get("signal_marker") or "")
                or receipt.get("match_policy")
                != "exact_clause_after_whitespace_normalization"
            ):
                raise OfferCatalogAdmissionError(
                    "Offer catalog admission commercial signal receipt is incomplete"
                )
            _require_sha256(
                str(receipt.get("source_sha256") or ""),
                field="commercial_signal.source_sha256",
            )
            _require_sha256(
                str(receipt.get("signal_sha256") or ""),
                field="commercial_signal.signal_sha256",
            )
            signal_excerpt = str(receipt.get("signal_excerpt") or "")
            if (
                not signal_excerpt
                or _sha256_text(signal_excerpt)
                != str(receipt.get("signal_sha256") or "")
            ):
                raise OfferCatalogAdmissionError(
                    "Offer catalog admission commercial signal excerpt is invalid"
                )
        if not self.reason_codes:
            raise OfferCatalogAdmissionError(
                "Offer catalog admission must include reason codes"
            )
        if artifact_digest(self._body()) != self.admission_digest:
            raise OfferCatalogAdmissionError("Offer catalog admission digest mismatch")
        if self.status is OfferCatalogAdmissionStatus.OFFERS_PRESENT:
            if (
                self.accepted_offer_count == 0
                or self.site_kind is not ZeroOfferSiteKind.NOT_APPLICABLE
            ):
                raise OfferCatalogAdmissionError(
                    "Offers-present admission has inconsistent site kind or count"
                )
        elif self.status is OfferCatalogAdmissionStatus.ZERO_OFFERS_ADMITTED:
            if self.accepted_offer_count != 0 or self.site_kind not in {
                ZeroOfferSiteKind.NONCOMMERCIAL,
                ZeroOfferSiteKind.NO_PUBLIC_OFFER,
            }:
                raise OfferCatalogAdmissionError(
                    "Zero-offer admission has inconsistent site kind or count"
                )
            if not self.source_evidence_receipts:
                raise OfferCatalogAdmissionError(
                    "Zero-offer admission has no source-bound evidence receipt"
                )
            if self.source_commercial_signal_receipts:
                raise OfferCatalogAdmissionError(
                    "Zero-offer admission conflicts with commercial source signals"
                )
            if (
                not self.source_absence_claims_allowed
                or self.crawl_coverage_state != "complete"
            ):
                raise OfferCatalogAdmissionError(
                    "Zero-offer admission lacks complete absence-safe crawl coverage"
                )
        elif (
            self.accepted_offer_count != 0
            or self.site_kind is not ZeroOfferSiteKind.COMMERCIAL_OR_UNKNOWN
        ):
            raise OfferCatalogAdmissionError(
                "Blocked zero-offer admission has inconsistent site kind or count"
            )

    def require_allowed(self) -> None:
        self.validate()
        if not self.allowed:
            raise OfferCatalogAdmissionError(
                "Zero accepted offers are not admissible for a commercial or "
                "unclassified site; generic INTENT generation is blocked "
                f"({', '.join(self.reason_codes)})"
            )


def assess_offer_catalog_admission(
    *,
    catalog: "OfferCatalog",
    profile: Mapping[str, Any],
    source_units: Iterable[SourceUnit | Mapping[str, Any]] = (),
    source_absence_claims_allowed: bool = False,
    crawl_coverage_state: str = "unknown",
) -> OfferCatalogAdmission:
    """Assess whether market research and INTENT generation may start."""

    catalog.validate()
    if not isinstance(profile, Mapping):
        raise OfferCatalogAdmissionError("Site profile must be a mapping")

    accepted_offer_count = len(catalog.accepted_offers)
    sources = tuple(
        raw if isinstance(raw, SourceUnit) else SourceUnit.from_mapping(raw)
        for raw in source_units
    )
    site_type = str(profile.get("site_type") or "").strip()
    business_model = str(profile.get("business_model") or "").strip()
    confidence = str(profile.get("confidence") or "").strip().casefold()
    evidence = _profile_model_evidence(profile)
    all_source_receipts, source_manifest_digest = (
        _source_bound_profile_evidence_receipts(
            evidence=evidence,
            source_units=sources,
        )
    )
    source_manifest_matches = (
        source_manifest_digest == catalog.source_manifest_digest
    )
    if not source_manifest_matches:
        all_source_receipts = ()
    combined_kind = " ".join(value for value in (site_type, business_model) if value)
    explicit_noncommercial = _contains_site_kind_marker(
        combined_kind,
        _NONCOMMERCIAL_SITE_KIND_MARKERS,
    )
    explicit_no_public_offer = _contains_site_kind_marker(
        combined_kind,
        _NO_PUBLIC_OFFER_SITE_KIND_MARKERS,
    )
    claimed_zero_site_kind = (
        ZeroOfferSiteKind.NONCOMMERCIAL
        if explicit_noncommercial
        else (
            ZeroOfferSiteKind.NO_PUBLIC_OFFER
            if explicit_no_public_offer
            else ZeroOfferSiteKind.COMMERCIAL_OR_UNKNOWN
        )
    )
    source_receipts = _site_kind_evidence_receipts(
        all_source_receipts,
        site_kind=claimed_zero_site_kind,
    )
    source_commercial_signal_receipts = _source_commercial_signal_receipts(
        sources
    )
    profile_commercial_signals = _profile_has_commercial_offer_signals(profile)
    source_commercial_signals = bool(source_commercial_signal_receipts)
    absence_contract_complete = bool(
        source_absence_claims_allowed is True
        and crawl_coverage_state == "complete"
    )
    evidence_present = bool(evidence)
    source_bound_site_kind_evidence_present = bool(source_receipts)
    confidence_sufficient = confidence in {"high", "medium"}

    if accepted_offer_count:
        status = OfferCatalogAdmissionStatus.OFFERS_PRESENT
        site_kind = ZeroOfferSiteKind.NOT_APPLICABLE
        reasons = ("accepted_offers_present",)
    elif (
        (explicit_noncommercial or explicit_no_public_offer)
        and source_bound_site_kind_evidence_present
        and confidence_sufficient
        and not profile_commercial_signals
        and not source_commercial_signals
        and absence_contract_complete
    ):
        status = OfferCatalogAdmissionStatus.ZERO_OFFERS_ADMITTED
        site_kind = (
            ZeroOfferSiteKind.NONCOMMERCIAL
            if explicit_noncommercial
            else ZeroOfferSiteKind.NO_PUBLIC_OFFER
        )
        reasons = (
            "explicit_noncommercial_site_kind"
            if explicit_noncommercial
            else "explicit_no_public_offer_site_kind",
            "source_bound_site_kind_evidence_present",
            "model_confidence_sufficient",
            "no_profile_commercial_offer_signals",
            "no_source_commercial_offer_signals",
            "complete_absence_safe_crawl_coverage",
        )
    else:
        status = OfferCatalogAdmissionStatus.ZERO_OFFERS_BLOCKED
        site_kind = ZeroOfferSiteKind.COMMERCIAL_OR_UNKNOWN
        reason_items: list[str] = []
        if not explicit_noncommercial and not explicit_no_public_offer:
            reason_items.append("missing_explicit_noncommercial_site_kind")
        if not evidence_present:
            reason_items.append("missing_model_evidence")
        elif not source_bound_site_kind_evidence_present:
            reason_items.append("missing_source_bound_model_evidence")
            reason_items.append("missing_source_bound_site_kind_evidence")
        if not source_manifest_matches:
            reason_items.append("source_manifest_mismatch")
        if not confidence_sufficient:
            reason_items.append("insufficient_model_confidence")
        if profile_commercial_signals:
            reason_items.append("commercial_offer_signals_present")
            reason_items.append("profile_commercial_offer_signals_present")
        if source_commercial_signals:
            if "commercial_offer_signals_present" not in reason_items:
                reason_items.append("commercial_offer_signals_present")
            reason_items.append("source_commercial_offer_signals_present")
        if source_absence_claims_allowed is not True:
            reason_items.append("source_absence_claims_not_allowed")
        if crawl_coverage_state != "complete":
            reason_items.append("crawl_coverage_not_complete")
        reasons = tuple(reason_items) or ("zero_offer_admission_not_proven",)

    body = {
        "version": OFFER_CATALOG_ADMISSION_VERSION,
        "status": status.value,
        "site_kind": site_kind.value,
        "catalog_digest": catalog.catalog_digest,
        "profile_digest": artifact_digest(profile),
        "accepted_offer_count": accepted_offer_count,
        "site_type": site_type,
        "business_model": business_model,
        "model_confidence": confidence,
        "model_evidence_count": len(evidence),
        "model_evidence_digest": artifact_digest(list(evidence)),
        "source_manifest_digest": source_manifest_digest,
        "source_evidence_receipts": list(source_receipts),
        "source_evidence_receipt_digest": artifact_digest(
            list(source_receipts)
        ),
        "source_commercial_signal_receipts": list(
            source_commercial_signal_receipts
        ),
        "source_commercial_signal_receipt_digest": artifact_digest(
            list(source_commercial_signal_receipts)
        ),
        "source_absence_claims_allowed": (
            source_absence_claims_allowed is True
        ),
        "crawl_coverage_state": str(crawl_coverage_state or "unknown"),
        "reason_codes": list(reasons),
    }
    admission = OfferCatalogAdmission(
        status=status,
        site_kind=site_kind,
        catalog_digest=catalog.catalog_digest,
        profile_digest=body["profile_digest"],
        accepted_offer_count=accepted_offer_count,
        site_type=site_type,
        business_model=business_model,
        model_confidence=confidence,
        model_evidence_count=len(evidence),
        model_evidence_digest=body["model_evidence_digest"],
        source_manifest_digest=source_manifest_digest,
        source_evidence_receipts=source_receipts,
        source_evidence_receipt_digest=body[
            "source_evidence_receipt_digest"
        ],
        source_commercial_signal_receipts=(
            source_commercial_signal_receipts
        ),
        source_commercial_signal_receipt_digest=body[
            "source_commercial_signal_receipt_digest"
        ],
        source_absence_claims_allowed=body[
            "source_absence_claims_allowed"
        ],
        crawl_coverage_state=body["crawl_coverage_state"],
        reason_codes=reasons,
        admission_digest=artifact_digest(body),
    )
    admission.validate()
    return admission


def _normalize_domain(value: str) -> str:
    raw = value.strip()
    if not raw:
        raise OfferCatalogError("client_domain must not be empty")
    if "://" not in raw:
        raw = f"https://{raw}"
    parsed = urlsplit(raw)
    host = (parsed.hostname or "").rstrip(".").casefold()
    if host.startswith("www."):
        host = host[4:]
    if not host:
        raise OfferCatalogError("client_domain must contain a valid host")
    return host


_NON_IDENTITY_DOMAIN_LABELS = frozenset(
    {
        "aiv",
        "app",
        "blog",
        "co",
        "com",
        "corp",
        "dev",
        "io",
        "net",
        "online",
        "org",
        "plus",
        "ru",
        "site",
        "shop",
        "store",
        "web",
        "www",
    }
)

_OFFLINE_SUFFIX_EXTRACTOR = tldextract.TLDExtract(
    suffix_list_urls=(),
    include_psl_private_domains=False,
)


def _registrable_identity_label(domain: str) -> str:
    """Return only the eTLD+1 registrable label from the bundled PSL."""

    extracted = _OFFLINE_SUFFIX_EXTRACTOR(domain)
    label = str(extracted.domain or "").strip().casefold()
    if not _normalize_phrase(label).replace(" ", ""):
        raise OfferCatalogError("client_domain has no registrable identity label")
    return label

def _source_bound_identity_receipt(
    alias: str,
    source: SourceUnit,
    *,
    client_domain: str,
) -> dict[str, Any] | None:
    """Prove a non-domain alias as the site's own identity.

    A first-party page is necessary but not sufficient: it can discuss a
    competitor.  Admission therefore requires an explicit identity statement,
    not mere co-occurrence or a commercial action.  The accepted rules are
    deliberately narrow and replayable from the exact, hashed page text.
    """

    if not _is_client_owned_url(source.source_url, client_domain):
        return None
    normalized_alias = _normalize_phrase(alias)
    normalized_text = _normalize_phrase(source.text)
    if not normalized_alias or not _contains_normalized_phrase(source.text, alias):
        return None
    alias_pattern = rf"(?<!\S){re.escape(normalized_alias)}(?!\S)"
    rules: tuple[tuple[str, str], ...] = (
        (
            "explicit_official_site_identity",
            rf"(?<!\S)(?:official\s+(?:web)?site(?:\s+of)?|"
            rf"официальн\w*\s+сайт(?:\s+(?:компани\w*|бренд\w*|"
            rf"агентств\w*|проект\w*))?)\s+{alias_pattern}",
        ),
        (
            "explicit_first_person_identity",
            rf"(?<!\S)(?:we\s+are|мы\s+(?:это|являемся)?)\s*"
            rf"{alias_pattern}(?!\S)",
        ),
    )
    for rule, pattern in rules:
        match = re.search(pattern, normalized_text)
        if match is not None:
            return {
                "source_unit_id": source.source_unit_id,
                "source_url": source.source_url,
                "source_sha256": source.source_sha256,
                "rule": rule,
                "normalized_claim": match.group(0),
                "normalized_claim_sha256": _sha256_text(match.group(0)),
            }

    return None


def _structured_identity_receipt(
    alias: str,
    record: Mapping[str, Any],
    *,
    source_by_id: Mapping[str, SourceUnit],
    client_domain: str,
) -> dict[str, Any] | None:
    """Bind a body identity claim to a code-owned homepage title segment."""

    source_unit_id = str(record.get("source_unit_id") or "").strip()
    source = source_by_id.get(source_unit_id)
    if source is None:
        return None
    source_url = str(record.get("source_url") or "").strip()
    source_sha256 = str(record.get("source_sha256") or "").strip()
    page_kind = str(record.get("page_kind") or "").strip().casefold()
    title = str(record.get("title") or "").strip()
    if (
        page_kind != "home"
        or not title
        or _normalize_url(source_url) != source.source_url
        or source_sha256 != source.source_sha256
        or not _is_client_owned_url(source.source_url, client_domain)
    ):
        return None
    # The first visual/title segment is the site identity; later segments are
    # descriptors and may contain partners or competitors.  Hyphen is not a
    # separator because it frequently belongs to a brand name.
    leading_title = re.split(r"\s*(?:\||—|–|·|:)\s*", title, maxsplit=1)[0]
    alias_compact = _normalize_phrase(alias).replace(" ", "")
    title_compact = _normalize_phrase(leading_title).replace(" ", "")
    if not alias_compact or title_compact != alias_compact:
        return None
    body_receipt = _source_bound_identity_receipt(
        alias,
        source,
        client_domain=client_domain,
    )
    if body_receipt is None:
        return None
    core = {
        "source_unit_id": source.source_unit_id,
        "source_url": source.source_url,
        "source_sha256": source.source_sha256,
        "page_kind": page_kind,
        "title": title,
        "title_sha256": _sha256_text(title),
        "leading_title_segment": leading_title,
        "leading_title_segment_sha256": _sha256_text(leading_title),
        "body_identity_receipt": body_receipt,
        "rule": "homepage_title_identity_plus_literal_body_claim",
    }
    return {**core, "receipt_sha256": artifact_digest(core)}


def admit_client_identity_aliases(
    *,
    client_domain: str,
    requested_aliases: Sequence[str],
    source_units: Iterable[SourceUnit | Mapping[str, Any]],
    structured_identity_records: Iterable[Mapping[str, Any]] = (),
) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Derive identity from the domain or a two-source first-party receipt."""

    domain = _normalize_domain(client_domain)
    registrable_label = _registrable_identity_label(domain)
    identity_labels = tuple(
        label
        for label in (registrable_label,)
        if _normalize_phrase(label) not in _NON_IDENTITY_DOMAIN_LABELS
        and len(_normalize_phrase(label).replace(" ", "")) >= 2
    )
    code_owned = _normalized_unique((domain, *identity_labels))
    identity_compacts = {
        _normalize_phrase(value).replace(" ", "") for value in code_owned
    }
    parsed_sources = tuple(
        raw if isinstance(raw, SourceUnit) else SourceUnit.from_mapping(raw)
        for raw in source_units
    )
    source_by_id = {source.source_unit_id: source for source in parsed_sources}
    structured_records = tuple(dict(record) for record in structured_identity_records)
    requested = _normalized_unique(requested_aliases)
    admitted_requested: list[str] = []
    rejected_requested: list[str] = []
    alias_receipts: list[dict[str, Any]] = []
    for alias in requested:
        compact = _normalize_phrase(alias).replace(" ", "")
        domain_derived = bool(compact and compact in identity_compacts)
        source_identity_receipts = [
            receipt
            for record in structured_records
            if (
                receipt := _structured_identity_receipt(
                    alias,
                    record,
                    source_by_id=source_by_id,
                    client_domain=domain,
                )
            )
            is not None
        ]
        if not domain_derived and not source_identity_receipts:
            rejected_requested.append(alias)
            continue
        admitted_requested.append(alias)
        occurrences = [
            {
                "source_unit_id": source.source_unit_id,
                "source_url": source.source_url,
                "source_sha256": source.source_sha256,
            }
            for source in parsed_sources
            if _is_client_owned_url(source.source_url, domain)
            and _contains_normalized_phrase(source.text, alias)
        ]
        alias_receipts.append(
            {
                "alias": alias,
                "alias_sha256": _sha256_text(_normalize_phrase(alias)),
                "derivation": (
                    "exact_compact_match_to_target_domain_label"
                    if domain_derived
                    else "exact_first_party_identity_statement"
                ),
                "source_occurrences": occurrences,
                "source_identity_receipts": source_identity_receipts,
            }
        )
    trusted = _normalized_unique((*code_owned, *admitted_requested))
    core = {
        "version": CLIENT_IDENTITY_ADMISSION_VERSION,
        "client_domain": domain,
        "domain_identity_labels": list(identity_labels),
        "requested_aliases": list(requested),
        "trusted_aliases": list(trusted),
        "rejected_aliases": list(rejected_requested),
        "alias_receipts": alias_receipts,
        "structured_identity_record_digest": artifact_digest(
            list(structured_records)
        ),
        "source_manifest_digest": artifact_digest(
            [source.descriptor() for source in sorted(
                parsed_sources,
                key=lambda item: item.source_unit_id,
            )]
        ),
    }
    return trusted, {
        **core,
        "admission_sha256": artifact_digest(core),
    }


def _normalize_url(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OfferCatalogError("source_url must not be empty")
    parsed = urlsplit(value.strip())
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.hostname:
        raise OfferCatalogError("source_url must be an absolute HTTP(S) URL")
    hostname = parsed.hostname.casefold().rstrip(".")
    port = parsed.port
    netloc = hostname
    if port and not (
        (parsed.scheme.casefold() == "http" and port == 80)
        or (parsed.scheme.casefold() == "https" and port == 443)
    ):
        netloc = f"{hostname}:{port}"
    path = parsed.path or "/"
    return urlunsplit((parsed.scheme.casefold(), netloc, path, parsed.query, ""))


def _url_domain(value: str) -> str:
    host = (urlsplit(value).hostname or "").casefold().rstrip(".")
    return host[4:] if host.startswith("www.") else host


def _is_client_owned_url(source_url: str, client_domain: str) -> bool:
    host = _url_domain(source_url)
    return host == client_domain or host.endswith(f".{client_domain}")


def _contains_normalized_phrase(text: str, phrase: str) -> bool:
    normalized_text = f" {_normalize_phrase(text)} "
    normalized_phrase = _normalize_phrase(phrase)
    return bool(normalized_phrase) and f" {normalized_phrase} " in normalized_text


def _user_job_bound_to_offer_excerpt(
    excerpt: str,
    *,
    canonical_name: str,
    user_job: str,
) -> bool:
    """Require the offer and customer job in the same exact source clause."""

    return any(
        _contains_normalized_phrase(clause, canonical_name)
        and _contains_normalized_phrase(clause, user_job)
        for clause in _OWNERSHIP_CLAUSE_SPLIT_RE.split(excerpt)
        if clause.strip()
    )


def _alias_identity_proven(
    evidence_excerpt: str,
    *,
    canonical_name: str,
    alias: str,
) -> bool:
    """Require literal, local evidence that two names denote one offer."""

    if not _contains_normalized_phrase(evidence_excerpt, canonical_name):
        return False
    if not _contains_normalized_phrase(evidence_excerpt, alias):
        return False
    # A short form embedded in the exact canonical name ("Compass" in
    # "AdTech Compass") is a deterministic alias.  A generic service word in a
    # different clause is not.
    if _contains_normalized_phrase(
        canonical_name,
        alias,
    ) or _contains_normalized_phrase(alias, canonical_name):
        return True

    canonical = re.escape(_normalize_phrase(canonical_name))
    normalized_alias = re.escape(_normalize_phrase(alias))
    excerpt = _normalize_phrase(evidence_excerpt)
    connector = (
        r"(?:aka|also known as|also called|"
        r"также извест\w* как|также называ\w*|сокращенн\w*|далее|или)"
    )
    return bool(
        re.search(
            rf"(?<!\S){canonical}(?!\S)\s+{connector}\s+"
            rf"(?<!\S){normalized_alias}(?!\S)",
            excerpt,
        )
        or re.search(
            rf"(?<!\S){normalized_alias}(?!\S)\s+{connector}\s+"
            rf"(?<!\S){canonical}(?!\S)",
            excerpt,
        )
    )


def _is_generic_offer_name(value: str) -> bool:
    return _normalize_phrase(value) in _GENERIC_OFFER_TERMS


def is_generic_offer_name(value: str) -> bool:
    """Expose the deterministic generic-category predicate to consumers."""

    return _is_generic_offer_name(value)


def _phrase_occurrences(text: str, values: Iterable[str]) -> tuple[tuple[int, int], ...]:
    """Return exact token-boundary spans in already normalized text."""

    spans: set[tuple[int, int]] = set()
    for value in values:
        phrase = _normalize_phrase(value)
        if not phrase:
            continue
        pattern = re.compile(rf"(?<!\S){re.escape(phrase)}(?!\S)")
        spans.update(match.span() for match in pattern.finditer(text))
    return tuple(sorted(spans))


def _first_person_actor_values(clause: str) -> tuple[str, ...]:
    """Return exact, structurally bounded first-party actor phrases."""

    return _normalized_unique(
        (
            *_FIRST_PERSON_ACTORS,
            *(match.group(0) for match in _FIRST_PERSON_ORGANIZATION_ACTOR_RE.finditer(clause)),
        )
    )


def _only_safe_actor_modifiers(value: str) -> bool:
    tokens = value.split()
    return len(tokens) <= 2 and all(
        token in _OWNERSHIP_SAFE_MODIFIERS for token in tokens
    )


def _structural_offer_object_prefix(value: str) -> bool:
    """Allow only grammatical offer introducers, never an arbitrary window."""

    tokens = value.split()
    return len(tokens) <= 5 and all(
        _OWNERSHIP_OBJECT_PREFIX_TOKEN_RE.fullmatch(token) is not None
        for token in tokens
    )


def _structural_coordinated_offer_prefix(value: str) -> bool:
    """Allow one bounded preceding object joined to the candidate offer.

    In ``create custom tattoos and precise piercings`` the second offer has a
    three-token prefix.  Requiring a final conjunction and at most four
    preceding tokens keeps this a local object-list rule instead of an
    arbitrary proximity window.
    """

    tokens = value.split()
    if not 2 <= len(tokens) <= 5:
        return False
    if tokens[-1] not in _COORDINATED_OFFER_CONJUNCTIONS:
        return False
    forbidden = {
        "but",
        "not",
        "without",
        "however",
        "no",
        "ne",
        "nije",
        "nisu",
        "без",
        "не",
        "нет",
        "однако",
        "но",
    }
    return not any(token in forbidden for token in tokens[:-1])


def _has_foreign_post_offer_attribution(
    clause: str,
    *,
    offer_end: int,
    client_aliases: Sequence[str],
) -> bool:
    suffix_tokens = clause[offer_end:].split()
    if not suffix_tokens or suffix_tokens[0] not in _FOREIGN_ATTRIBUTION_MARKERS:
        return False
    attribution = " ".join(suffix_tokens[:6])
    return not bool(_phrase_occurrences(attribution, client_aliases))


def _direct_actor_action_offer_binding(
    clause: str,
    *,
    actor_values: Sequence[str],
    offer_values: Sequence[str],
    client_aliases: Sequence[str],
) -> bool:
    """Prove a grammatical actor -> commercial action -> offer chain.

    Actor adjacency and the absence of another commercial action before the
    candidate are structural boundaries.  A mere actor/offer co-occurrence in
    the same sentence, or an arbitrary character-distance window, is not
    sufficient.
    """

    actors = _phrase_occurrences(clause, actor_values)
    offers = _phrase_occurrences(clause, offer_values)
    actions = tuple(_OWNERSHIP_ACTION_RE.finditer(clause))
    for _actor_start, actor_end in actors:
        for action_index, action in enumerate(actions):
            if action.start() < actor_end:
                continue
            if not _only_safe_actor_modifiers(clause[actor_end : action.start()]):
                continue
            next_action_start = (
                actions[action_index + 1].start()
                if action_index + 1 < len(actions)
                else len(clause) + 1
            )
            for offer_start, offer_end in offers:
                if not action.end() <= offer_start < next_action_start:
                    continue
                if not _structural_offer_object_prefix(
                    clause[action.end() : offer_start]
                ):
                    continue
                if _has_foreign_post_offer_attribution(
                    clause,
                    offer_end=offer_end,
                    client_aliases=client_aliases,
                ):
                    continue
                return True
    return False


def _first_party_staff_delivery_binding(
    clause: str,
    *,
    actor_values: Sequence[str],
    offer_values: Sequence[str],
    client_aliases: Sequence[str],
) -> bool:
    """Prove a first-party actor -> offered setting -> staff delivery chain.

    Marketing copy often places the concrete service in a relative clause:
    ``our studio offers an environment where artists create custom tattoos``.
    Treating the whole sentence as a proximity window would be unsafe.  This
    helper instead requires both action edges, an allowlisted grammatical
    bridge, and a direct or coordinated object of the staff delivery action.
    """

    actors = _phrase_occurrences(clause, actor_values)
    scope_actions = tuple(_FIRST_PARTY_SCOPE_ACTION_RE.finditer(clause))
    delivery_actions = tuple(_OWNERSHIP_ACTION_RE.finditer(clause))
    offers = _phrase_occurrences(clause, offer_values)
    for _actor_start, actor_end in actors:
        for scope_action in scope_actions:
            if scope_action.start() < actor_end:
                continue
            if not _only_safe_actor_modifiers(
                clause[actor_end : scope_action.start()]
            ):
                continue
            for action_index, delivery_action in enumerate(delivery_actions):
                if delivery_action.start() <= scope_action.end():
                    continue
                bridge = clause[scope_action.end() : delivery_action.start()].strip()
                if _FIRST_PARTY_STAFF_BRIDGE_RE.fullmatch(bridge) is None:
                    continue
                next_action_start = (
                    delivery_actions[action_index + 1].start()
                    if action_index + 1 < len(delivery_actions)
                    else len(clause) + 1
                )
                for offer_start, offer_end in offers:
                    if not delivery_action.end() <= offer_start < next_action_start:
                        continue
                    prefix = clause[delivery_action.end() : offer_start]
                    if not (
                        _structural_offer_object_prefix(prefix)
                        or _structural_coordinated_offer_prefix(prefix)
                    ):
                        continue
                    if _has_foreign_post_offer_attribution(
                        clause,
                        offer_end=offer_end,
                        client_aliases=client_aliases,
                    ):
                        continue
                    return True
    return False


def _copular_or_possessive_offer_binding(
    clause: str,
    *,
    actor_values: Sequence[str],
    offer_values: Sequence[str],
) -> bool:
    """Recognize explicit ownership statements without an action verb."""

    actors = tuple(
        re.escape(_normalize_phrase(value))
        for value in actor_values
        if _normalize_phrase(value)
    )
    offers = tuple(
        re.escape(_normalize_phrase(value))
        for value in offer_values
        if _normalize_phrase(value)
    )
    if not actors or not offers:
        return False
    actor = rf"(?:{'|'.join(actors)})"
    offer = rf"(?:{'|'.join(offers)})"
    boundary = r"(?<!\S){}(?!\S)"
    connector = rf"(?:\s+{_OWNERSHIP_CONNECTOR_PATTERN}){{0,3}}"
    patterns = (
        # Compass — продукт Realweb / Compass is a product of Example.
        rf"{boundary.format(offer)}{connector}\s+{_OWNERSHIP_TYPE_PATTERN}"
        rf"{connector}\s+{boundary.format(actor)}",
        # Compass is a Realweb product.
        rf"{boundary.format(offer)}{connector}\s+{boundary.format(actor)}"
        rf"\s+{_OWNERSHIP_TYPE_PATTERN}",
        # Realweb product Compass / Realweb's Compass platform.
        rf"{boundary.format(actor)}(?:\s+s)?{connector}\s+"
        rf"(?:{_OWNERSHIP_TYPE_PATTERN}\s+)?{boundary.format(offer)}"
        rf"(?:\s+{_OWNERSHIP_TYPE_PATTERN})?",
        # продукт Realweb Compass / платформа Compass от Realweb.
        rf"(?<!\S){_OWNERSHIP_TYPE_PATTERN}\s+{boundary.format(actor)}"
        rf"\s+{boundary.format(offer)}",
        rf"(?<!\S){_OWNERSHIP_TYPE_PATTERN}\s+{boundary.format(offer)}"
        rf"(?:\s+(?:от|by|of))\s+{boundary.format(actor)}",
    )
    return any(re.search(pattern, clause) is not None for pattern in patterns)


def _client_offer_binding_proven(
    excerpt: str,
    *,
    offer_names: Sequence[str],
    client_aliases: Sequence[str],
    source_is_client_owned: bool,
) -> bool:
    """Require an explicit actor-to-offer relation in the exact excerpt.

    A first-party URL establishes who owns the page, not every noun mentioned
    on it.  It only licenses unquoted first-person ownership forms.  Named
    client aliases work on any source, but must be the actor or possessor of
    the candidate offer inside one grammatical clause.
    """

    clauses = tuple(
        normalized
        for raw_clause in _OWNERSHIP_CLAUSE_SPLIT_RE.split(excerpt)
        if (normalized := _normalize_phrase(raw_clause))
    )
    for clause in clauses:
        if not _phrase_occurrences(clause, offer_names):
            continue
        if _direct_actor_action_offer_binding(
            clause,
            actor_values=client_aliases,
            offer_values=offer_names,
            client_aliases=client_aliases,
        ) or _copular_or_possessive_offer_binding(
            clause,
            actor_values=client_aliases,
            offer_values=offer_names,
        ):
            return True

        # On a client-owned source, first-person language is an explicit actor
        # only outside quoted/attributed speech.  A competitor quote such as
        # `Rival: «мы предлагаем SEO»` must not become a client offer.
        if source_is_client_owned and not any(
            marker in excerpt for marker in _QUOTE_MARKERS
        ):
            first_person_actors = _first_person_actor_values(clause)
            if _direct_actor_action_offer_binding(
                clause,
                actor_values=first_person_actors,
                offer_values=offer_names,
                client_aliases=client_aliases,
            ) or _first_party_staff_delivery_binding(
                clause,
                actor_values=first_person_actors,
                offer_values=offer_names,
                client_aliases=client_aliases,
            ) or _copular_or_possessive_offer_binding(
                clause,
                actor_values=_FIRST_PERSON_POSSESSIVES,
                offer_values=offer_names,
            ):
                return True
    return False


@dataclass(frozen=True, slots=True)
class SourceUnit:
    """Exact text and identity of one crawled or researched source."""

    source_unit_id: str
    source_url: str
    text: str
    source_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.source_unit_id, str) or not self.source_unit_id.strip():
            raise OfferEvidenceError("source_unit_id must not be empty")
        if not isinstance(self.text, str):
            raise OfferEvidenceError("SourceUnit.text must be a string")
        normalized_url = _normalize_url(self.source_url)
        digest = _require_sha256(self.source_sha256, field="source_sha256")
        if _sha256_text(self.text) != digest:
            raise OfferEvidenceError(
                f"Source unit {self.source_unit_id!r} text digest mismatch"
            )
        object.__setattr__(self, "source_unit_id", self.source_unit_id.strip())
        object.__setattr__(self, "source_url", normalized_url)

    @classmethod
    def from_text(cls, *, source_unit_id: str, source_url: str, text: str) -> "SourceUnit":
        return cls(
            source_unit_id=source_unit_id,
            source_url=source_url,
            text=text,
            source_sha256=_sha256_text(text),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SourceUnit":
        return cls(
            source_unit_id=str(value.get("source_unit_id", "")),
            source_url=str(value.get("source_url", "")),
            text=value.get("text", ""),
            source_sha256=str(value.get("source_sha256", "")),
        )

    def descriptor(self) -> dict[str, str | int]:
        return {
            "source_unit_id": self.source_unit_id,
            "source_url": self.source_url,
            "source_sha256": self.source_sha256,
            "utf8_length": len(self.text.encode("utf-8")),
        }


@dataclass(frozen=True, slots=True)
class OfferCandidate:
    """Probabilistic extractor output awaiting deterministic admission."""

    canonical_name: str
    aliases: tuple[str, ...]
    kind: OfferKind
    source_url: str
    evidence_excerpt: str
    source_unit_id: str
    source_sha256: str
    confidence: float
    user_jobs: tuple[str, ...]
    commercially_relevant: bool = True

    def __post_init__(self) -> None:
        name = _SPACE_RE.sub(" ", self.canonical_name).strip()
        if not name:
            raise OfferCatalogError("canonical_name must not be empty")
        if not isinstance(self.evidence_excerpt, str) or not self.evidence_excerpt:
            raise OfferEvidenceError("evidence_excerpt must preserve non-empty exact text")
        if not isinstance(self.source_unit_id, str) or not self.source_unit_id.strip():
            raise OfferEvidenceError("source_unit_id must not be empty")
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)):
            raise OfferCatalogError("confidence must be a finite number from 0 to 1")
        confidence = float(self.confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise OfferCatalogError("confidence must be a finite number from 0 to 1")
        try:
            kind = self.kind if isinstance(self.kind, OfferKind) else OfferKind(self.kind)
        except ValueError as exc:
            raise OfferCatalogError("kind must be product, service, or direction") from exc
        if isinstance(self.aliases, str) or not isinstance(self.aliases, Sequence):
            raise OfferCatalogError("aliases must be an array of strings")
        if isinstance(self.user_jobs, str) or not isinstance(self.user_jobs, Sequence):
            raise OfferCatalogError("user_jobs must be an array of strings")
        aliases = _normalized_unique(self.aliases)
        jobs = _normalized_unique(self.user_jobs)
        object.__setattr__(self, "canonical_name", name)
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "source_url", _normalize_url(self.source_url))
        object.__setattr__(self, "source_unit_id", self.source_unit_id.strip())
        object.__setattr__(
            self, "source_sha256", _require_sha256(self.source_sha256, field="source_sha256")
        )
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "user_jobs", jobs)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "OfferCandidate":
        aliases = value.get("aliases", ())
        jobs = value.get("user_jobs", ())
        if isinstance(aliases, str) or not isinstance(aliases, Sequence):
            raise OfferCatalogError("aliases must be an array of strings")
        if isinstance(jobs, str) or not isinstance(jobs, Sequence):
            raise OfferCatalogError("user_jobs must be an array of strings")
        try:
            kind = OfferKind(str(value.get("kind", "")))
        except ValueError as exc:
            raise OfferCatalogError("kind must be product, service, or direction") from exc
        return cls(
            canonical_name=str(value.get("canonical_name", "")),
            aliases=tuple(aliases),
            kind=kind,
            source_url=str(value.get("source_url", "")),
            evidence_excerpt=value.get("evidence_excerpt", ""),
            source_unit_id=str(value.get("source_unit_id", "")),
            source_sha256=str(value.get("source_sha256", "")),
            confidence=value.get("confidence", float("nan")),
            user_jobs=tuple(jobs),
            commercially_relevant=value.get("commercially_relevant", True) is True,
        )

    @property
    def candidate_id(self) -> str:
        identity = {
            "canonical_name": self.canonical_name,
            "aliases": list(self.aliases),
            "kind": self.kind.value,
            "source_url": self.source_url,
            "evidence_excerpt_sha256": _sha256_text(self.evidence_excerpt),
            "source_unit_id": self.source_unit_id,
            "source_sha256": self.source_sha256,
            "confidence": self.confidence,
            "user_jobs": list(self.user_jobs),
            "commercially_relevant": self.commercially_relevant,
        }
        return f"candidate:{artifact_digest(identity)}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "canonical_name": self.canonical_name,
            "aliases": list(self.aliases),
            "kind": self.kind.value,
            "source_url": self.source_url,
            "evidence_excerpt": self.evidence_excerpt,
            "source_unit_id": self.source_unit_id,
            "source_sha256": self.source_sha256,
            "confidence": self.confidence,
            "user_jobs": list(self.user_jobs),
            "commercially_relevant": self.commercially_relevant,
        }


@dataclass(frozen=True, slots=True)
class OfferEvidenceRef:
    source_url: str
    evidence_excerpt: str
    source_unit_id: str
    source_sha256: str
    evidence_sha256: str
    client_binding_proven: bool
    version: str = OFFER_EVIDENCE_VERSION

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "OfferEvidenceRef":
        excerpt = value.get("evidence_excerpt", "")
        if not isinstance(excerpt, str):
            raise OfferEvidenceError("evidence_excerpt must be a string")
        evidence_sha256 = str(value.get("evidence_sha256", ""))
        _require_sha256(evidence_sha256, field="evidence_sha256")
        if _sha256_text(excerpt) != evidence_sha256:
            raise OfferEvidenceError("evidence excerpt digest mismatch")
        source_sha256 = str(value.get("source_sha256", ""))
        _require_sha256(source_sha256, field="source_sha256")
        version = str(value.get("version", ""))
        if version != OFFER_EVIDENCE_VERSION:
            raise OfferEvidenceError("Unsupported offer evidence version")
        return cls(
            source_url=_normalize_url(str(value.get("source_url", ""))),
            evidence_excerpt=excerpt,
            source_unit_id=str(value.get("source_unit_id", "")).strip(),
            source_sha256=source_sha256,
            evidence_sha256=evidence_sha256,
            client_binding_proven=value.get("client_binding_proven") is True,
            version=version,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "source_url": self.source_url,
            "evidence_excerpt": self.evidence_excerpt,
            "source_unit_id": self.source_unit_id,
            "source_sha256": self.source_sha256,
            "evidence_sha256": self.evidence_sha256,
            "client_binding_proven": self.client_binding_proven,
        }


@dataclass(frozen=True, slots=True)
class OfferUserJobEvidence:
    """Exact first-party evidence for one customer job used in scenarios."""

    user_job: str
    source_url: str
    evidence_excerpt: str
    source_unit_id: str
    source_sha256: str
    evidence_sha256: str
    user_job_sha256: str
    version: str = OFFER_USER_JOB_EVIDENCE_VERSION

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "OfferUserJobEvidence":
        user_job = str(value.get("user_job") or "").strip()
        evidence_excerpt = value.get("evidence_excerpt", "")
        if not user_job or not isinstance(evidence_excerpt, str):
            raise OfferEvidenceError("Offer user-job evidence is incomplete")
        source_sha256 = _require_sha256(
            str(value.get("source_sha256") or ""),
            field="source_sha256",
        )
        evidence_sha256 = _require_sha256(
            str(value.get("evidence_sha256") or ""),
            field="evidence_sha256",
        )
        user_job_sha256 = _require_sha256(
            str(value.get("user_job_sha256") or ""),
            field="user_job_sha256",
        )
        if _sha256_text(evidence_excerpt) != evidence_sha256:
            raise OfferEvidenceError("Offer user-job excerpt digest mismatch")
        if _sha256_text(_normalize_phrase(user_job)) != user_job_sha256:
            raise OfferEvidenceError("Offer user-job digest mismatch")
        if not _contains_normalized_phrase(evidence_excerpt, user_job):
            raise OfferEvidenceError("Offer user job is not literal in evidence")
        if str(value.get("version") or "") != OFFER_USER_JOB_EVIDENCE_VERSION:
            raise OfferEvidenceError("Unsupported offer user-job evidence version")
        return cls(
            user_job=user_job,
            source_url=_normalize_url(str(value.get("source_url") or "")),
            evidence_excerpt=evidence_excerpt,
            source_unit_id=str(value.get("source_unit_id") or "").strip(),
            source_sha256=source_sha256,
            evidence_sha256=evidence_sha256,
            user_job_sha256=user_job_sha256,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "user_job": self.user_job,
            "source_url": self.source_url,
            "evidence_excerpt": self.evidence_excerpt,
            "source_unit_id": self.source_unit_id,
            "source_sha256": self.source_sha256,
            "evidence_sha256": self.evidence_sha256,
            "user_job_sha256": self.user_job_sha256,
        }


@dataclass(frozen=True, slots=True)
class AcceptedOffer:
    offer_id: str
    canonical_name: str
    aliases: tuple[str, ...]
    kind: OfferKind
    source_url: str
    evidence_excerpt: str
    source_unit_id: str
    source_sha256: str
    confidence: float
    user_jobs: tuple[str, ...]
    user_job_evidence: tuple[OfferUserJobEvidence, ...]
    evidence_refs: tuple[OfferEvidenceRef, ...]
    generic_category_term: bool

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AcceptedOffer":
        aliases = value.get("aliases", ())
        jobs = value.get("user_jobs", ())
        raw_refs = value.get("evidence_refs", ())
        raw_job_evidence = value.get("user_job_evidence", ())
        if isinstance(aliases, str) or not isinstance(aliases, Sequence):
            raise OfferCatalogError("Accepted offer aliases must be an array")
        if isinstance(jobs, str) or not isinstance(jobs, Sequence):
            raise OfferCatalogError("Accepted offer user_jobs must be an array")
        if isinstance(raw_refs, (str, bytes)) or not isinstance(raw_refs, Sequence):
            raise OfferCatalogError("Accepted offer evidence_refs must be an array")
        if isinstance(raw_job_evidence, (str, bytes)) or not isinstance(
            raw_job_evidence,
            Sequence,
        ):
            raise OfferCatalogError(
                "Accepted offer user_job_evidence must be an array"
            )
        try:
            kind = OfferKind(str(value.get("kind", "")))
        except ValueError as exc:
            raise OfferCatalogError("Accepted offer kind is invalid") from exc
        return cls(
            offer_id=str(value.get("offer_id", "")),
            canonical_name=str(value.get("canonical_name", "")),
            aliases=tuple(str(item) for item in aliases),
            kind=kind,
            source_url=_normalize_url(str(value.get("source_url", ""))),
            evidence_excerpt=value.get("evidence_excerpt", ""),
            source_unit_id=str(value.get("source_unit_id", "")),
            source_sha256=str(value.get("source_sha256", "")),
            confidence=float(value.get("confidence", float("nan"))),
            user_jobs=tuple(str(item) for item in jobs),
            user_job_evidence=tuple(
                OfferUserJobEvidence.from_mapping(item)
                for item in raw_job_evidence
            ),
            evidence_refs=tuple(OfferEvidenceRef.from_mapping(item) for item in raw_refs),
            generic_category_term=value.get("generic_category_term") is True,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "offer_id": self.offer_id,
            "canonical_name": self.canonical_name,
            "aliases": list(self.aliases),
            "kind": self.kind.value,
            "source_url": self.source_url,
            "evidence_excerpt": self.evidence_excerpt,
            "source_unit_id": self.source_unit_id,
            "source_sha256": self.source_sha256,
            "confidence": self.confidence,
            "user_jobs": list(self.user_jobs),
            "user_job_evidence": [
                item.as_dict() for item in self.user_job_evidence
            ],
            "evidence_refs": [item.as_dict() for item in self.evidence_refs],
            "generic_category_term": self.generic_category_term,
        }


@dataclass(frozen=True, slots=True)
class OfferDisposition:
    candidate_id: str
    canonical_name: str
    decision: DispositionDecision
    reason: str
    accepted_offer_id: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "OfferDisposition":
        try:
            decision = DispositionDecision(str(value.get("decision", "")))
        except ValueError as exc:
            raise OfferCatalogError("Offer disposition decision is invalid") from exc
        accepted_offer_id = value.get("accepted_offer_id")
        if accepted_offer_id is not None:
            accepted_offer_id = str(accepted_offer_id)
        return cls(
            candidate_id=str(value.get("candidate_id", "")),
            canonical_name=str(value.get("canonical_name", "")),
            decision=decision,
            reason=str(value.get("reason", "")),
            accepted_offer_id=accepted_offer_id,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "canonical_name": self.canonical_name,
            "decision": self.decision.value,
            "reason": self.reason,
            "accepted_offer_id": self.accepted_offer_id,
        }


@dataclass(frozen=True, slots=True)
class OfferCatalog:
    client_domain: str
    client_aliases: tuple[str, ...]
    accepted_offers: tuple[AcceptedOffer, ...]
    dispositions: tuple[OfferDisposition, ...]
    source_manifest_digest: str
    catalog_digest: str
    version: str = OFFER_CATALOG_VERSION

    def _body(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "client_domain": self.client_domain,
            "client_aliases": list(self.client_aliases),
            "accepted_offers": [item.as_dict() for item in self.accepted_offers],
            "dispositions": [item.as_dict() for item in self.dispositions],
            "source_manifest_digest": self.source_manifest_digest,
        }

    def as_dict(self) -> dict[str, Any]:
        value = self._body()
        value["catalog_digest"] = self.catalog_digest
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "OfferCatalog":
        raw_offers = value.get("accepted_offers", ())
        raw_dispositions = value.get("dispositions", ())
        raw_aliases = value.get("client_aliases", ())
        if isinstance(raw_offers, (str, bytes)) or not isinstance(raw_offers, Sequence):
            raise OfferCatalogError("accepted_offers must be an array")
        if isinstance(raw_dispositions, (str, bytes)) or not isinstance(
            raw_dispositions, Sequence
        ):
            raise OfferCatalogError("dispositions must be an array")
        if isinstance(raw_aliases, str) or not isinstance(raw_aliases, Sequence):
            raise OfferCatalogError("client_aliases must be an array")
        version = str(value.get("version", ""))
        if version != OFFER_CATALOG_VERSION:
            raise OfferCatalogError("Unsupported offer catalog version")
        catalog = cls(
            client_domain=_normalize_domain(str(value.get("client_domain", ""))),
            client_aliases=tuple(str(item) for item in raw_aliases),
            accepted_offers=tuple(AcceptedOffer.from_mapping(item) for item in raw_offers),
            dispositions=tuple(
                OfferDisposition.from_mapping(item) for item in raw_dispositions
            ),
            source_manifest_digest=str(value.get("source_manifest_digest", "")),
            catalog_digest=str(value.get("catalog_digest", "")),
            version=version,
        )
        catalog.validate()
        return catalog

    def validate(self) -> None:
        if len(self.accepted_offers) > MAX_ACCEPTED_OFFERS:
            raise OfferCatalogError("Catalog exceeds the hard maximum of 10 offers")
        _require_sha256(self.source_manifest_digest, field="source_manifest_digest")
        _require_sha256(self.catalog_digest, field="catalog_digest")
        if artifact_digest(self._body()) != self.catalog_digest:
            raise OfferCatalogError("catalog_digest does not match catalog content")
        offer_ids = [item.offer_id for item in self.accepted_offers]
        if len(set(offer_ids)) != len(offer_ids):
            raise OfferCatalogError("Accepted offer IDs must be unique")
        candidate_ids = [item.candidate_id for item in self.dispositions]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise OfferCatalogError("Offer disposition candidate IDs must be unique")
        accepted_ids = set(offer_ids)
        for offer in self.accepted_offers:
            if not offer.canonical_name.strip() or not offer.evidence_refs:
                raise OfferCatalogError("Accepted offer lacks a name or evidence")
            expected_offer_id = f"offer:{artifact_digest({
                'canonical_name': offer.canonical_name,
                'aliases': list(offer.aliases),
                'kind': offer.kind.value,
            })}"
            if offer.offer_id != expected_offer_id:
                raise OfferCatalogError("Accepted offer_id is invalid")
            if offer.generic_category_term is not _is_generic_offer_name(
                offer.canonical_name
            ):
                raise OfferCatalogError(
                    "Accepted offer generic-category flag is inconsistent"
                )
            if (
                isinstance(offer.confidence, bool)
                or not math.isfinite(offer.confidence)
                or not 0 <= offer.confidence <= 1
            ):
                raise OfferCatalogError("Accepted offer confidence is invalid")
            _require_sha256(offer.source_sha256, field="source_sha256")
            matching_primary = [
                evidence
                for evidence in offer.evidence_refs
                if evidence.source_url == offer.source_url
                and evidence.source_unit_id == offer.source_unit_id
                and evidence.source_sha256 == offer.source_sha256
                and evidence.evidence_excerpt == offer.evidence_excerpt
            ]
            if not matching_primary:
                raise OfferCatalogError("Accepted offer primary evidence is not in evidence_refs")
            for evidence in offer.evidence_refs:
                _require_sha256(evidence.source_sha256, field="source_sha256")
                _require_sha256(evidence.evidence_sha256, field="evidence_sha256")
                if _sha256_text(evidence.evidence_excerpt) != evidence.evidence_sha256:
                    raise OfferCatalogError("Accepted offer evidence digest mismatch")
                if not evidence.client_binding_proven:
                    raise OfferCatalogError(
                        "Accepted offer evidence lacks explicit client ownership"
                    )
            if not any(
                _contains_normalized_phrase(
                    evidence.evidence_excerpt,
                    offer.canonical_name,
                )
                and _client_offer_binding_proven(
                    evidence.evidence_excerpt,
                    offer_names=(offer.canonical_name,),
                    client_aliases=self.client_aliases,
                    source_is_client_owned=_is_client_owned_url(
                        evidence.source_url,
                        self.client_domain,
                    ),
                )
                for evidence in offer.evidence_refs
            ):
                raise OfferCatalogError(
                    "Accepted offer canonical name lacks client-bound literal evidence"
                )
            for alias in offer.aliases:
                if not any(
                    not (
                        _is_generic_offer_name(alias)
                        and not _is_generic_offer_name(offer.canonical_name)
                    )
                    and _alias_identity_proven(
                        evidence.evidence_excerpt,
                        canonical_name=offer.canonical_name,
                        alias=alias,
                    )
                    and _client_offer_binding_proven(
                        evidence.evidence_excerpt,
                        offer_names=(offer.canonical_name,),
                        client_aliases=self.client_aliases,
                        source_is_client_owned=_is_client_owned_url(
                            evidence.source_url,
                            self.client_domain,
                        ),
                    )
                    for evidence in offer.evidence_refs
                ):
                    raise OfferCatalogError(
                        "Accepted offer alias lacks client-bound literal evidence"
                    )
            receipt_jobs = _normalized_unique(
                receipt.user_job for receipt in offer.user_job_evidence
            )
            if receipt_jobs != _normalized_unique(offer.user_jobs):
                raise OfferCatalogError(
                    "Accepted offer user jobs lack exact evidence coverage"
                )
            for receipt in offer.user_job_evidence:
                if not _user_job_bound_to_offer_excerpt(
                    receipt.evidence_excerpt,
                    canonical_name=offer.canonical_name,
                    user_job=receipt.user_job,
                ):
                    raise OfferCatalogError(
                        "Accepted offer user job is not bound to the offer clause"
                    )
                if not any(
                    evidence.source_url == receipt.source_url
                    and evidence.source_unit_id == receipt.source_unit_id
                    and evidence.source_sha256 == receipt.source_sha256
                    and evidence.evidence_sha256 == receipt.evidence_sha256
                    and evidence.evidence_excerpt == receipt.evidence_excerpt
                    for evidence in offer.evidence_refs
                ):
                    raise OfferCatalogError(
                        "Accepted offer user-job receipt is outside offer evidence"
                    )
            if offer.generic_category_term and offer.kind is OfferKind.PRODUCT:
                raise OfferCatalogError("Generic category term cannot be an accepted product")
        for disposition in self.dispositions:
            if not disposition.candidate_id.startswith("candidate:"):
                raise OfferCatalogError("Disposition candidate_id is invalid")
            if disposition.decision in {
                DispositionDecision.ACCEPTED,
                DispositionDecision.DUPLICATE,
            } and disposition.accepted_offer_id not in accepted_ids:
                raise OfferCatalogError("Disposition references an unknown accepted offer")
            if disposition.decision in {
                DispositionDecision.REJECTED,
                DispositionDecision.OVERFLOW,
            } and disposition.accepted_offer_id is not None:
                raise OfferCatalogError("Rejected or overflow disposition cannot bind an offer")

    def legacy_product_strings(self) -> tuple[str, ...]:
        """Derived compatibility view; never an independent source of truth."""

        return tuple(item.canonical_name for item in self.accepted_offers)


@dataclass(frozen=True, slots=True)
class _AdmittedCandidate:
    candidate: OfferCandidate
    evidence: OfferEvidenceRef
    effective_canonical_name: str
    canonical_promoted_from_alias: bool
    generic: bool
    effective_kind: OfferKind
    kind_normalized: bool
    grounded_aliases: tuple[str, ...]
    aliases_pruned: bool
    grounded_user_jobs: tuple[str, ...]
    user_job_evidence: tuple[OfferUserJobEvidence, ...]
    user_jobs_pruned: bool
    identity_keys: frozenset[str]


def normalize_offer_candidates_against_sources(
    *,
    source_units: Iterable[SourceUnit | Mapping[str, Any]],
    candidates: Iterable[OfferCandidate | Mapping[str, Any]],
) -> tuple[tuple[OfferCandidate, ...], dict[str, Any]]:
    """Repair only code-provable source coordinates and audit malformed rows.

    An extractor can copy a URL or digest incorrectly even when its literal
    evidence excerpt is valid.  The code may repair those coordinates only
    when the excerpt identifies exactly one saved source unit.  It never
    rewrites names, excerpts, kinds, confidence or customer jobs.  Malformed
    rows are excluded with a content-addressed audit record so that one bad
    candidate cannot discard the rest of the proven catalog.
    """

    source_map: dict[str, SourceUnit] = {}
    for raw_source in source_units:
        source = (
            raw_source
            if isinstance(raw_source, SourceUnit)
            else SourceUnit.from_mapping(raw_source)
        )
        previous = source_map.get(source.source_unit_id)
        if previous is not None and previous != source:
            raise OfferEvidenceError(
                f"Conflicting source units use ID {source.source_unit_id!r}"
            )
        source_map[source.source_unit_id] = source
    sources = tuple(source_map[key] for key in sorted(source_map))

    normalized: list[OfferCandidate] = []
    rows: list[dict[str, Any]] = []
    for ordinal, raw_candidate in enumerate(candidates, start=1):
        if isinstance(raw_candidate, OfferCandidate):
            value = raw_candidate.as_dict()
        elif isinstance(raw_candidate, Mapping):
            value = dict(raw_candidate)
        else:
            rows.append(
                {
                    "ordinal": ordinal,
                    "raw_candidate_sha256": artifact_digest(
                        {"python_type": type(raw_candidate).__name__}
                    ),
                    "canonical_name": "",
                    "status": "malformed",
                    "repairs": [],
                    "error": "candidate_must_be_an_object",
                }
            )
            continue
        raw_digest = artifact_digest(value)
        excerpt = value.get("evidence_excerpt")
        excerpt = excerpt if isinstance(excerpt, str) else ""
        declared_id = str(value.get("source_unit_id") or "").strip()
        declared_url = str(value.get("source_url") or "").strip()
        selected = source_map.get(declared_id)
        if selected is not None and excerpt not in selected.text:
            selected = None
        if selected is None and excerpt:
            url_matches = [
                source
                for source in sources
                if source.source_url == declared_url and excerpt in source.text
            ]
            if len(url_matches) == 1:
                selected = url_matches[0]
        if selected is None and excerpt:
            excerpt_matches = [
                source for source in sources if excerpt in source.text
            ]
            if len(excerpt_matches) == 1:
                selected = excerpt_matches[0]

        if selected is None:
            excerpt_match_count = sum(
                excerpt in source.text for source in sources
            ) if excerpt else 0
            rows.append(
                {
                    "ordinal": ordinal,
                    "raw_candidate_sha256": raw_digest,
                    "canonical_name": str(
                        value.get("canonical_name") or ""
                    ).strip(),
                    "status": "unresolved_evidence",
                    "repairs": [],
                    "error": (
                        "literal_excerpt_matches_multiple_source_units"
                        if excerpt_match_count > 1
                        else "literal_excerpt_does_not_resolve_to_saved_source"
                    ),
                }
            )
            continue

        repairs: list[str] = []
        exact_values = {
            "source_unit_id": selected.source_unit_id,
            "source_url": selected.source_url,
            "source_sha256": selected.source_sha256,
        }
        for field, exact in exact_values.items():
            if value.get(field) != exact:
                value[field] = exact
                repairs.append(field)
        try:
            candidate = OfferCandidate.from_mapping(value)
        except OfferCatalogError as exc:
            rows.append(
                {
                    "ordinal": ordinal,
                    "raw_candidate_sha256": raw_digest,
                    "canonical_name": str(
                        value.get("canonical_name") or ""
                    ).strip(),
                    "status": "malformed",
                    "repairs": sorted(repairs),
                    "error": str(exc),
                }
            )
            continue
        normalized.append(candidate)
        rows.append(
            {
                "ordinal": ordinal,
                "raw_candidate_sha256": raw_digest,
                "canonical_name": candidate.canonical_name,
                "status": "repaired" if repairs else "valid",
                "repairs": sorted(repairs),
                "error": None,
            }
        )

    manifest_core = {
        "version": OFFER_CANDIDATE_NORMALIZATION_VERSION,
        "source_manifest_digest": artifact_digest(
            [source.descriptor() for source in sources]
        ),
        "input_candidate_count": len(rows),
        "valid_candidate_count": len(normalized),
        "repaired_candidate_count": sum(
            row["status"] == "repaired" for row in rows
        ),
        "malformed_candidate_count": sum(
            row["status"] == "malformed" for row in rows
        ),
        "unresolved_evidence_count": sum(
            row["status"] == "unresolved_evidence" for row in rows
        ),
        "rows": rows,
    }
    return tuple(normalized), {
        **manifest_core,
        "manifest_sha256": artifact_digest(manifest_core),
    }


def _candidate_identity_keys(
    candidate: OfferCandidate,
    *,
    effective_canonical_name: str,
    grounded_aliases: Sequence[str],
) -> frozenset[str]:
    canonical = _normalize_phrase(effective_canonical_name)
    all_names = (effective_canonical_name, *grounded_aliases)
    distinctive = {
        _normalize_phrase(value)
        for value in all_names
        if _normalize_phrase(value) and not _is_generic_offer_name(value)
    }
    return frozenset(distinctive or {canonical})


def _validate_candidate_evidence(
    candidate: OfferCandidate,
    *,
    source_units: Mapping[str, SourceUnit],
    client_domain: str,
    client_aliases: Sequence[str],
) -> _AdmittedCandidate:
    source = source_units.get(candidate.source_unit_id)
    if source is None:
        raise OfferEvidenceError("unknown_source_unit")
    if source.source_sha256 != candidate.source_sha256:
        raise OfferEvidenceError("source_digest_mismatch")
    if source.source_url != candidate.source_url:
        raise OfferEvidenceError("source_url_mismatch")
    if candidate.evidence_excerpt not in source.text:
        raise OfferEvidenceError("excerpt_not_found_verbatim_in_source")

    canonical_is_literal = _contains_normalized_phrase(
        candidate.evidence_excerpt,
        candidate.canonical_name,
    )
    if not candidate.commercially_relevant:
        raise OfferEvidenceError("not_commercially_relevant")

    source_is_client_owned = _is_client_owned_url(source.source_url, client_domain)
    promoted_aliases = tuple(
        alias
        for alias in candidate.aliases
        if not canonical_is_literal
        and source_is_client_owned
        and not _is_generic_offer_name(alias)
        and _contains_normalized_phrase(candidate.evidence_excerpt, alias)
        and _client_offer_binding_proven(
            candidate.evidence_excerpt,
            offer_names=(alias,),
            client_aliases=client_aliases,
            source_is_client_owned=source_is_client_owned,
        )
    )
    if canonical_is_literal:
        effective_canonical_name = candidate.canonical_name
        canonical_promoted_from_alias = False
    elif promoted_aliases:
        # ``OfferCandidate.aliases`` is normalized into a stable lexical order,
        # so the promotion is deterministic.  Crucially, the model's
        # translated or normalized name is not retained as an alias: only the
        # literal first-party phrase becomes the evidence-facing identity.
        effective_canonical_name = promoted_aliases[0]
        canonical_promoted_from_alias = True
    else:
        raise OfferEvidenceError("canonical_offer_name_not_literal_in_excerpt")

    generic = _is_generic_offer_name(effective_canonical_name)
    client_binding_proven = _client_offer_binding_proven(
        candidate.evidence_excerpt,
        offer_names=(effective_canonical_name,),
        client_aliases=client_aliases,
        source_is_client_owned=source_is_client_owned,
    )
    grounded_aliases = tuple(
        alias
        for alias in candidate.aliases
        if _normalize_phrase(alias) != _normalize_phrase(effective_canonical_name)
        if _contains_normalized_phrase(candidate.evidence_excerpt, alias)
        and not (
            _is_generic_offer_name(alias)
            and not generic
        )
        and client_binding_proven
        and _alias_identity_proven(
            candidate.evidence_excerpt,
            canonical_name=effective_canonical_name,
            alias=alias,
        )
    )
    grounded_user_jobs = tuple(
        job
        for job in candidate.user_jobs
        if _user_job_bound_to_offer_excerpt(
            candidate.evidence_excerpt,
            canonical_name=effective_canonical_name,
            user_job=job,
        )
    )

    if generic and candidate.kind is OfferKind.PRODUCT and not client_binding_proven:
        raise OfferEvidenceError("generic_category_term_is_not_a_proprietary_product")
    if generic and not client_binding_proven:
        raise OfferEvidenceError("generic_category_term_without_client_offer_binding")
    if not client_binding_proven:
        raise OfferEvidenceError("offer_without_explicit_client_ownership_binding")

    # A generic category can be a proven client service, but it must never be
    # emitted as a proprietary product merely because an extractor chose the
    # wrong enum.  Correct the taxonomy while preserving the original
    # candidate and disposition in the audit trail.
    effective_kind = (
        OfferKind.SERVICE
        if generic and candidate.kind is OfferKind.PRODUCT
        else candidate.kind
    )

    evidence = OfferEvidenceRef(
        source_url=source.source_url,
        evidence_excerpt=candidate.evidence_excerpt,
        source_unit_id=source.source_unit_id,
        source_sha256=source.source_sha256,
        evidence_sha256=_sha256_text(candidate.evidence_excerpt),
        client_binding_proven=client_binding_proven,
    )
    user_job_evidence = tuple(
        OfferUserJobEvidence(
            user_job=job,
            source_url=source.source_url,
            evidence_excerpt=candidate.evidence_excerpt,
            source_unit_id=source.source_unit_id,
            source_sha256=source.source_sha256,
            evidence_sha256=evidence.evidence_sha256,
            user_job_sha256=_sha256_text(_normalize_phrase(job)),
        )
        for job in grounded_user_jobs
    )
    return _AdmittedCandidate(
        candidate=candidate,
        evidence=evidence,
        effective_canonical_name=effective_canonical_name,
        canonical_promoted_from_alias=canonical_promoted_from_alias,
        generic=generic,
        effective_kind=effective_kind,
        kind_normalized=effective_kind is not candidate.kind,
        grounded_aliases=grounded_aliases,
        aliases_pruned=len(grounded_aliases) != len(candidate.aliases),
        grounded_user_jobs=grounded_user_jobs,
        user_job_evidence=user_job_evidence,
        user_jobs_pruned=len(grounded_user_jobs) != len(candidate.user_jobs),
        identity_keys=_candidate_identity_keys(
            candidate,
            effective_canonical_name=effective_canonical_name,
            grounded_aliases=grounded_aliases,
        ),
    )


def _candidate_preference(item: _AdmittedCandidate) -> tuple[Any, ...]:
    candidate = item.candidate
    return (
        -candidate.confidence,
        -int(item.evidence.client_binding_proven),
        -len(item.grounded_user_jobs),
        int(item.generic),
        _normalize_phrase(item.effective_canonical_name),
        item.effective_kind.value,
        candidate.source_unit_id,
        candidate.candidate_id,
    )


def _componentize(candidates: Sequence[_AdmittedCandidate]) -> list[list[_AdmittedCandidate]]:
    if not candidates:
        return []
    parents = list(range(len(candidates)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            parents[right_root] = left_root
        else:
            parents[left_root] = right_root

    for left_index, left in enumerate(candidates):
        left_canonical = _normalize_phrase(left.effective_canonical_name)
        for right_index in range(left_index + 1, len(candidates)):
            right = candidates[right_index]
            same_canonical = left_canonical == _normalize_phrase(
                right.effective_canonical_name
            )
            if same_canonical or left.identity_keys.intersection(right.identity_keys):
                union(left_index, right_index)

    groups: dict[int, list[_AdmittedCandidate]] = {}
    for index, item in enumerate(candidates):
        groups.setdefault(find(index), []).append(item)
    components = [sorted(group, key=_candidate_preference) for group in groups.values()]
    return sorted(components, key=lambda group: _candidate_preference(group[0]))


def _accepted_offer(component: Sequence[_AdmittedCandidate]) -> AcceptedOffer:
    representative = min(component, key=_candidate_preference)
    candidate = representative.candidate
    aliases = _normalized_unique(
        value
        for item in component
        for value in (item.effective_canonical_name, *item.grounded_aliases)
        if _normalize_phrase(value)
        != _normalize_phrase(representative.effective_canonical_name)
    )
    jobs = _normalized_unique(
        job for item in component for job in item.grounded_user_jobs
    )
    user_job_evidence = tuple(
        sorted(
            {
                (
                    _normalize_phrase(receipt.user_job),
                    receipt.evidence_sha256,
                ): receipt
                for item in component
                for receipt in item.user_job_evidence
            }.values(),
            key=lambda item: (
                _normalize_phrase(item.user_job),
                item.source_unit_id,
                item.evidence_sha256,
            ),
        )
    )
    evidence_refs = tuple(
        sorted(
            {item.evidence.evidence_sha256: item.evidence for item in component}.values(),
            key=lambda item: (
                item.source_unit_id,
                item.source_url,
                item.evidence_sha256,
            ),
        )
    )
    primary = representative.evidence
    semantic_identity = {
        "canonical_name": representative.effective_canonical_name,
        "aliases": list(aliases),
        "kind": representative.effective_kind.value,
    }
    return AcceptedOffer(
        offer_id=f"offer:{artifact_digest(semantic_identity)}",
        canonical_name=representative.effective_canonical_name,
        aliases=aliases,
        kind=representative.effective_kind,
        source_url=primary.source_url,
        evidence_excerpt=primary.evidence_excerpt,
        source_unit_id=primary.source_unit_id,
        source_sha256=primary.source_sha256,
        confidence=candidate.confidence,
        user_jobs=jobs,
        user_job_evidence=user_job_evidence,
        evidence_refs=evidence_refs,
        generic_category_term=representative.generic,
    )


def build_offer_catalog(
    *,
    client_domain: str,
    client_aliases: Sequence[str],
    source_units: Iterable[SourceUnit | Mapping[str, Any]],
    candidates: Iterable[OfferCandidate | Mapping[str, Any]],
    max_offers: int = MAX_ACCEPTED_OFFERS,
) -> OfferCatalog:
    """Admit at most ten proven offers and disposition every candidate.

    Invalid evidence rejects that candidate without discarding other proven
    offers.  Malformed candidate mappings still fail visibly because they have
    no trustworthy identity from which to create an audit disposition.
    """

    if isinstance(max_offers, bool) or not isinstance(max_offers, int):
        raise OfferCatalogError("max_offers must be an integer from 1 to 10")
    if not 1 <= max_offers <= MAX_ACCEPTED_OFFERS:
        raise OfferCatalogError("max_offers must be an integer from 1 to 10")
    domain = _normalize_domain(client_domain)
    aliases = _normalized_unique(client_aliases)
    if not aliases:
        aliases = (domain,)

    source_map: dict[str, SourceUnit] = {}
    for raw_source in source_units:
        source = (
            raw_source
            if isinstance(raw_source, SourceUnit)
            else SourceUnit.from_mapping(raw_source)
        )
        existing = source_map.get(source.source_unit_id)
        if existing is not None and existing != source:
            raise OfferEvidenceError(
                f"Conflicting source units use ID {source.source_unit_id!r}"
            )
        source_map[source.source_unit_id] = source
    source_manifest = [
        source_map[key].descriptor() for key in sorted(source_map)
    ]
    source_manifest_digest = artifact_digest(source_manifest)

    parsed_candidates = [
        raw if isinstance(raw, OfferCandidate) else OfferCandidate.from_mapping(raw)
        for raw in candidates
    ]
    parsed_candidates = list(
        {item.candidate_id: item for item in parsed_candidates}.values()
    )
    parsed_candidates.sort(key=lambda item: item.candidate_id)
    admitted: list[_AdmittedCandidate] = []
    dispositions: list[OfferDisposition] = []
    for candidate in parsed_candidates:
        try:
            admitted.append(
                _validate_candidate_evidence(
                    candidate,
                    source_units=source_map,
                    client_domain=domain,
                    client_aliases=aliases,
                )
            )
        except OfferEvidenceError as exc:
            dispositions.append(
                OfferDisposition(
                    candidate_id=candidate.candidate_id,
                    canonical_name=candidate.canonical_name,
                    decision=DispositionDecision.REJECTED,
                    reason=str(exc),
                )
            )

    components = _componentize(admitted)
    accepted: list[AcceptedOffer] = []
    for component_index, component in enumerate(components):
        offer = _accepted_offer(component)
        within_scope = component_index < max_offers
        if within_scope:
            accepted.append(offer)
        representative = min(component, key=_candidate_preference)
        for item in component:
            if not within_scope:
                decision = DispositionDecision.OVERFLOW
                reason = f"outside_commercial_offer_scope_max_{max_offers}"
                accepted_offer_id = None
            elif item.candidate.candidate_id == representative.candidate.candidate_id:
                decision = DispositionDecision.ACCEPTED
                reason = (
                    "source_bound_commercial_offer_canonical_promoted_from_literal_alias"
                    if item.canonical_promoted_from_alias
                    else (
                        "source_bound_commercial_offer_kind_normalized_to_service"
                        if item.kind_normalized
                        else (
                            "source_bound_commercial_offer_ungrounded_aliases_removed"
                            if item.aliases_pruned
                            else (
                                "source_bound_commercial_offer_ungrounded_user_jobs_removed"
                                if item.user_jobs_pruned
                                else "source_bound_commercial_offer"
                            )
                        )
                    )
                )
                accepted_offer_id = offer.offer_id
            else:
                decision = DispositionDecision.DUPLICATE
                reason = "merged_by_canonical_name_or_distinctive_alias"
                accepted_offer_id = offer.offer_id
            dispositions.append(
                OfferDisposition(
                    candidate_id=item.candidate.candidate_id,
                    canonical_name=item.candidate.canonical_name,
                    decision=decision,
                    reason=reason,
                    accepted_offer_id=accepted_offer_id,
                )
            )

    accepted_tuple = tuple(accepted)
    disposition_tuple = tuple(
        sorted(dispositions, key=lambda item: (item.candidate_id, item.decision.value))
    )
    body = {
        "version": OFFER_CATALOG_VERSION,
        "client_domain": domain,
        "client_aliases": list(aliases),
        "accepted_offers": [item.as_dict() for item in accepted_tuple],
        "dispositions": [item.as_dict() for item in disposition_tuple],
        "source_manifest_digest": source_manifest_digest,
    }
    catalog = OfferCatalog(
        client_domain=domain,
        client_aliases=aliases,
        accepted_offers=accepted_tuple,
        dispositions=disposition_tuple,
        source_manifest_digest=source_manifest_digest,
        catalog_digest=artifact_digest(body),
    )
    catalog.validate()
    return catalog


@dataclass(frozen=True, slots=True)
class PayloadShard:
    index: int
    count: int
    utf8_offset: int
    utf8_length: int
    text: str
    shard_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "count": self.count,
            "utf8_offset": self.utf8_offset,
            "utf8_length": self.utf8_length,
            "text": self.text,
            "shard_sha256": self.shard_sha256,
        }


@dataclass(frozen=True, slots=True)
class DomainResearchPayload:
    document: dict[str, Any]
    shards: tuple[PayloadShard, ...]
    manifest: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "document": json.loads(_canonical_json(self.document)),
            "shards": [item.as_dict() for item in self.shards],
            "manifest": json.loads(_canonical_json(self.manifest)),
        }


def _split_utf8_losslessly(text: str, target_utf8_bytes: int) -> list[str]:
    if isinstance(target_utf8_bytes, bool) or not isinstance(target_utf8_bytes, int):
        raise OfferCatalogError("target_utf8_bytes must be a positive integer")
    if target_utf8_bytes <= 0:
        raise OfferCatalogError("target_utf8_bytes must be a positive integer")
    if not text:
        return [""]
    result: list[str] = []
    buffer: list[str] = []
    byte_count = 0
    for character in text:
        width = len(character.encode("utf-8"))
        if buffer and byte_count + width > target_utf8_bytes:
            result.append("".join(buffer))
            buffer = []
            byte_count = 0
        buffer.append(character)
        byte_count += width
        if len(buffer) == 1 and byte_count > target_utf8_bytes:
            result.append(character)
            buffer = []
            byte_count = 0
    if buffer:
        result.append("".join(buffer))
    if "".join(result) != text:
        raise AssertionError("Internal lossless UTF-8 split mismatch")
    return result


def build_domain_research_payload(
    catalog: OfferCatalog,
    *,
    target_utf8_bytes: int = 16_384,
) -> DomainResearchPayload:
    """Create a compact, complete offer context and lossless shard manifest."""

    catalog.validate()
    document = {
        "version": DOMAIN_RESEARCH_PAYLOAD_VERSION,
        "client_domain": catalog.client_domain,
        "client_aliases": list(catalog.client_aliases),
        "catalog_digest": catalog.catalog_digest,
        "offer_count": len(catalog.accepted_offers),
        "offers": [
            {
                "offer_id": offer.offer_id,
                "canonical_name": offer.canonical_name,
                "aliases": list(offer.aliases),
                "kind": offer.kind.value,
                "confidence": offer.confidence,
                "user_jobs": list(offer.user_jobs),
                "evidence_refs": [item.as_dict() for item in offer.evidence_refs],
            }
            for offer in catalog.accepted_offers
        ],
    }
    serialized = _canonical_json(document)
    fragments = _split_utf8_losslessly(serialized, target_utf8_bytes)
    count = len(fragments)
    shards: list[PayloadShard] = []
    offset = 0
    for index, fragment in enumerate(fragments):
        length = len(fragment.encode("utf-8"))
        shards.append(
            PayloadShard(
                index=index,
                count=count,
                utf8_offset=offset,
                utf8_length=length,
                text=fragment,
                shard_sha256=_sha256_text(fragment),
            )
        )
        offset += length
    manifest_body = {
        "version": DOMAIN_RESEARCH_MANIFEST_VERSION,
        "payload_version": DOMAIN_RESEARCH_PAYLOAD_VERSION,
        "catalog_digest": catalog.catalog_digest,
        "document_sha256": _sha256_text(serialized),
        "document_utf8_length": len(serialized.encode("utf-8")),
        "offer_ids": [offer.offer_id for offer in catalog.accepted_offers],
        "offer_count": len(catalog.accepted_offers),
        "shard_count": count,
        "shards": [
            {
                "index": item.index,
                "utf8_offset": item.utf8_offset,
                "utf8_length": item.utf8_length,
                "shard_sha256": item.shard_sha256,
            }
            for item in shards
        ],
    }
    manifest = dict(manifest_body)
    manifest["manifest_sha256"] = artifact_digest(manifest_body)
    payload = DomainResearchPayload(
        document=document,
        shards=tuple(shards),
        manifest=manifest,
    )
    reconstruct_domain_research_payload(payload)
    return payload


def reconstruct_domain_research_payload(
    payload: DomainResearchPayload,
) -> dict[str, Any]:
    manifest = payload.manifest
    manifest_sha256 = manifest.get("manifest_sha256")
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if not isinstance(manifest_sha256, str) or artifact_digest(body) != manifest_sha256:
        raise OfferCatalogError("Domain research manifest digest mismatch")
    expected_count = manifest.get("shard_count")
    if expected_count != len(payload.shards):
        raise OfferCatalogError("Domain research shard count mismatch")
    offset = 0
    fragments: list[str] = []
    descriptors = manifest.get("shards")
    if not isinstance(descriptors, list) or len(descriptors) != len(payload.shards):
        raise OfferCatalogError("Domain research shard descriptors are incomplete")
    for index, (shard, descriptor) in enumerate(zip(payload.shards, descriptors, strict=True)):
        if shard.index != index or shard.count != len(payload.shards):
            raise OfferCatalogError("Domain research shards are not contiguous")
        if shard.utf8_offset != offset:
            raise OfferCatalogError("Domain research shard offsets contain a gap or overlap")
        actual_length = len(shard.text.encode("utf-8"))
        if actual_length != shard.utf8_length or _sha256_text(shard.text) != shard.shard_sha256:
            raise OfferCatalogError("Domain research shard content identity mismatch")
        expected_descriptor = {
            "index": shard.index,
            "utf8_offset": shard.utf8_offset,
            "utf8_length": shard.utf8_length,
            "shard_sha256": shard.shard_sha256,
        }
        if descriptor != expected_descriptor:
            raise OfferCatalogError("Domain research shard descriptor mismatch")
        offset += actual_length
        fragments.append(shard.text)
    serialized = "".join(fragments)
    if offset != manifest.get("document_utf8_length"):
        raise OfferCatalogError("Domain research document length mismatch")
    if _sha256_text(serialized) != manifest.get("document_sha256"):
        raise OfferCatalogError("Domain research document digest mismatch")
    try:
        reconstructed = json.loads(serialized)
    except json.JSONDecodeError as exc:
        raise OfferCatalogError("Domain research shards do not reconstruct JSON") from exc
    if _canonical_json(reconstructed) != _canonical_json(payload.document):
        raise OfferCatalogError("Domain research document differs from shard reconstruction")
    offer_ids = [item.get("offer_id") for item in reconstructed.get("offers", [])]
    if offer_ids != manifest.get("offer_ids"):
        raise OfferCatalogError("Domain research manifest does not cover every offer")
    return reconstructed


@dataclass(frozen=True, slots=True)
class OfferCluster:
    cluster_id: str
    offer_ids: tuple[str, ...]
    canonical_names: tuple[str, ...]
    user_jobs: tuple[str, ...]
    version: str = OFFER_CLUSTER_VERSION

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "OfferCluster":
        offer_ids = value.get("offer_ids", ())
        names = value.get("canonical_names", ())
        jobs = value.get("user_jobs", ())
        if any(
            isinstance(items, str) or not isinstance(items, Sequence)
            for items in (offer_ids, names, jobs)
        ):
            raise PromptCoverageError("Offer cluster arrays are invalid")
        version = str(value.get("version", ""))
        if version != OFFER_CLUSTER_VERSION:
            raise PromptCoverageError("Unsupported offer cluster version")
        return cls(
            cluster_id=str(value.get("cluster_id", "")),
            offer_ids=tuple(str(item) for item in offer_ids),
            canonical_names=tuple(str(item) for item in names),
            user_jobs=tuple(str(item) for item in jobs),
            version=version,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "cluster_id": self.cluster_id,
            "offer_ids": list(self.offer_ids),
            "canonical_names": list(self.canonical_names),
            "user_jobs": list(self.user_jobs),
        }


def build_offer_clusters(
    catalog: OfferCatalog,
    *,
    assignments: Mapping[str, Sequence[str]] | None = None,
    max_clusters: int = 6,
) -> tuple[OfferCluster, ...]:
    """Partition all accepted offers into at most six coverage clusters.

    Callers may provide reviewed semantic assignments.  Without them, offers
    sharing the same first normalized user job are grouped; lower-priority
    overflow groups are combined into the sixth coverage bucket.  No offer is
    omitted in either mode.
    """

    catalog.validate()
    if isinstance(max_clusters, bool) or not isinstance(max_clusters, int):
        raise PromptCoverageError("max_clusters must be an integer from 1 to 6")
    if not 1 <= max_clusters <= 6:
        raise PromptCoverageError("max_clusters must be an integer from 1 to 6")
    by_id = {offer.offer_id: offer for offer in catalog.accepted_offers}

    groups: list[tuple[str, list[str]]] = []
    if assignments is not None:
        seen: set[str] = set()
        for label in sorted(assignments, key=_normalize_phrase):
            raw_ids = assignments[label]
            if isinstance(raw_ids, str):
                raise PromptCoverageError("Cluster assignments must contain offer ID arrays")
            offer_ids = sorted(set(raw_ids))
            if not offer_ids:
                raise PromptCoverageError("A supplied offer cluster must not be empty")
            unknown = set(offer_ids).difference(by_id)
            duplicate = set(offer_ids).intersection(seen)
            if unknown:
                raise PromptCoverageError(f"Unknown offer IDs in cluster: {sorted(unknown)}")
            if duplicate:
                raise PromptCoverageError(
                    f"Offers occur in more than one cluster: {sorted(duplicate)}"
                )
            seen.update(offer_ids)
            groups.append((str(label), offer_ids))
        missing = set(by_id).difference(seen)
        if missing:
            raise PromptCoverageError(f"Cluster assignments omit offers: {sorted(missing)}")
        if len(groups) > max_clusters:
            raise PromptCoverageError("Supplied assignments exceed six offer clusters")
    else:
        automatic: dict[str, list[str]] = {}
        for offer in sorted(
            catalog.accepted_offers,
            key=lambda item: (-item.confidence, _normalize_phrase(item.canonical_name), item.offer_id),
        ):
            primary_job = (
                _normalize_phrase(offer.user_jobs[0])
                if offer.user_jobs
                else f"offer {offer.offer_id}"
            )
            automatic.setdefault(primary_job, []).append(offer.offer_id)
        ranked = sorted(
            automatic.items(),
            key=lambda item: (
                -max(by_id[offer_id].confidence for offer_id in item[1]),
                item[0],
            ),
        )
        if len(ranked) <= max_clusters:
            groups = [(label, sorted(ids)) for label, ids in ranked]
        else:
            groups = [(label, sorted(ids)) for label, ids in ranked[: max_clusters - 1]]
            overflow_ids = sorted(
                offer_id for _label, ids in ranked[max_clusters - 1 :] for offer_id in ids
            )
            groups.append(("combined coverage", overflow_ids))

    result: list[OfferCluster] = []
    for _label, offer_ids in groups:
        offers = [by_id[offer_id] for offer_id in sorted(offer_ids)]
        names = tuple(offer.canonical_name for offer in offers)
        jobs = _normalized_unique(job for offer in offers for job in offer.user_jobs)
        identity = {"offer_ids": [offer.offer_id for offer in offers]}
        result.append(
            OfferCluster(
                cluster_id=f"cluster:{artifact_digest(identity)}",
                offer_ids=tuple(offer.offer_id for offer in offers),
                canonical_names=names,
                user_jobs=jobs,
            )
        )
    result.sort(key=lambda item: item.cluster_id)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class IntentPrompt:
    prompt_key: str
    intent: str
    text: str
    supporting_cluster_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if isinstance(self.supporting_cluster_ids, str) or not isinstance(
            self.supporting_cluster_ids, Sequence
        ):
            raise PromptCoverageError("supporting_cluster_ids must be an array")
        key = self.prompt_key.strip()
        intent = self.intent.strip().upper()
        if not key or not isinstance(self.text, str) or not self.text.strip():
            raise PromptCoverageError("Every INTENT prompt needs a key and non-empty text")
        if intent not in INTENT_CODES:
            raise PromptCoverageError(f"Unknown INTENT code: {self.intent!r}")
        object.__setattr__(self, "prompt_key", key)
        object.__setattr__(self, "intent", intent)
        object.__setattr__(self, "supporting_cluster_ids", tuple(sorted(set(self.supporting_cluster_ids))))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "IntentPrompt":
        cluster_ids = value.get("supporting_cluster_ids", ())
        if isinstance(cluster_ids, str) or not isinstance(cluster_ids, Sequence):
            raise PromptCoverageError("supporting_cluster_ids must be an array")
        return cls(
            prompt_key=str(value.get("prompt_key", "")),
            intent=str(value.get("intent", "")),
            text=value.get("text", ""),
            supporting_cluster_ids=tuple(str(item) for item in cluster_ids),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt_key": self.prompt_key,
            "intent": self.intent,
            "text": self.text,
            "text_sha256": _sha256_text(self.text),
            "supporting_cluster_ids": list(self.supporting_cluster_ids),
        }


@dataclass(frozen=True, slots=True)
class ClusterExclusion:
    cluster_id: str
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.reason, str):
            raise PromptCoverageError("Cluster exclusion reason must be a string")
        reason = _SPACE_RE.sub(" ", self.reason).strip()
        vague = _normalize_phrase(reason) in {
            "n a",
            "na",
            "none",
            "not applicable",
            "нет",
            "не применимо",
            "прочее",
        }
        if not self.cluster_id.strip() or len(reason) < 12 or vague:
            raise PromptCoverageError(
                "Cluster exclusion needs a specific, auditable reason"
            )
        object.__setattr__(self, "cluster_id", self.cluster_id.strip())
        object.__setattr__(self, "reason", reason)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ClusterExclusion":
        return cls(
            cluster_id=str(value.get("cluster_id", "")),
            reason=value.get("reason", ""),
        )

    def as_dict(self) -> dict[str, str]:
        return {"cluster_id": self.cluster_id, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class UpstreamArtifactDigests:
    profile_digest: str
    catalog_digest: str
    market_research_digest: str
    selected_pages_digest: str
    version: str = UPSTREAM_DIGESTS_VERSION

    def __post_init__(self) -> None:
        for field, value in (
            ("profile_digest", self.profile_digest),
            ("catalog_digest", self.catalog_digest),
            ("market_research_digest", self.market_research_digest),
            ("selected_pages_digest", self.selected_pages_digest),
        ):
            _require_sha256(value, field=field)

    def as_dict(self) -> dict[str, str]:
        return {
            "version": self.version,
            "profile_digest": self.profile_digest,
            "catalog_digest": self.catalog_digest,
            "market_research_digest": self.market_research_digest,
            "selected_pages_digest": self.selected_pages_digest,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "UpstreamArtifactDigests":
        version = str(value.get("version", ""))
        if version != UPSTREAM_DIGESTS_VERSION:
            raise OfferCatalogError("Unsupported upstream artifact digest version")
        return cls(
            profile_digest=str(value.get("profile_digest", "")),
            catalog_digest=str(value.get("catalog_digest", "")),
            market_research_digest=str(value.get("market_research_digest", "")),
            selected_pages_digest=str(value.get("selected_pages_digest", "")),
            version=version,
        )


def build_upstream_artifact_digests(
    *,
    profile: Any,
    catalog: OfferCatalog,
    market_research: Any,
    selected_pages_manifest: Any,
) -> UpstreamArtifactDigests:
    catalog.validate()
    return UpstreamArtifactDigests(
        profile_digest=artifact_digest(profile),
        catalog_digest=catalog.catalog_digest,
        market_research_digest=artifact_digest(market_research),
        selected_pages_digest=artifact_digest(selected_pages_manifest),
    )


@dataclass(frozen=True, slots=True)
class PromptFoundation:
    upstream: UpstreamArtifactDigests
    clusters: tuple[OfferCluster, ...]
    prompts: tuple[IntentPrompt, ...]
    exclusions: tuple[ClusterExclusion, ...]
    coverage: tuple[dict[str, Any], ...]
    prompt_set_digest: str
    coverage_digest: str
    foundation_digest: str
    version: str = PROMPT_FOUNDATION_VERSION

    def _body(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "upstream": self.upstream.as_dict(),
            "clusters": [item.as_dict() for item in self.clusters],
            "prompts": [item.as_dict() for item in self.prompts],
            "exclusions": [item.as_dict() for item in self.exclusions],
            "coverage": [json.loads(_canonical_json(item)) for item in self.coverage],
            "prompt_set_digest": self.prompt_set_digest,
            "coverage_digest": self.coverage_digest,
        }

    def as_dict(self) -> dict[str, Any]:
        value = self._body()
        value["foundation_digest"] = self.foundation_digest
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PromptFoundation":
        version = str(value.get("version", ""))
        if version != PROMPT_FOUNDATION_VERSION:
            raise PromptCoverageError("Unsupported prompt foundation version")
        raw_clusters = value.get("clusters", ())
        raw_prompts = value.get("prompts", ())
        raw_exclusions = value.get("exclusions", ())
        raw_coverage = value.get("coverage", ())
        if any(
            isinstance(items, (str, bytes)) or not isinstance(items, Sequence)
            for items in (raw_clusters, raw_prompts, raw_exclusions, raw_coverage)
        ):
            raise PromptCoverageError("Prompt foundation arrays are invalid")
        upstream = value.get("upstream")
        if not isinstance(upstream, Mapping):
            raise PromptCoverageError("Prompt foundation upstream digests are missing")
        foundation = cls(
            upstream=UpstreamArtifactDigests.from_mapping(upstream),
            clusters=tuple(OfferCluster.from_mapping(item) for item in raw_clusters),
            prompts=tuple(IntentPrompt.from_mapping(item) for item in raw_prompts),
            exclusions=tuple(ClusterExclusion.from_mapping(item) for item in raw_exclusions),
            coverage=tuple(json.loads(_canonical_json(item)) for item in raw_coverage),
            prompt_set_digest=str(value.get("prompt_set_digest", "")),
            coverage_digest=str(value.get("coverage_digest", "")),
            foundation_digest=str(value.get("foundation_digest", "")),
            version=version,
        )
        foundation.validate()
        return foundation

    @property
    def prompt_keys(self) -> tuple[str, ...]:
        return tuple(item.prompt_key for item in self.prompts)

    def validate(self) -> None:
        _require_sha256(self.prompt_set_digest, field="prompt_set_digest")
        _require_sha256(self.coverage_digest, field="coverage_digest")
        _require_sha256(self.foundation_digest, field="foundation_digest")
        prompt_data = [item.as_dict() for item in self.prompts]
        if artifact_digest(prompt_data) != self.prompt_set_digest:
            raise PromptCoverageError("prompt_set_digest mismatch")
        if artifact_digest(list(self.coverage)) != self.coverage_digest:
            raise PromptCoverageError("coverage_digest mismatch")
        if artifact_digest(self._body()) != self.foundation_digest:
            raise PromptCoverageError("foundation_digest mismatch")
        if len(self.prompts) != len(INTENT_CODES):
            raise PromptCoverageError("Prompt foundation requires exactly six INTENT prompts")
        if {item.intent for item in self.prompts} != set(INTENT_CODES):
            raise PromptCoverageError("Prompt foundation requires each INTENT code exactly once")
        if len({item.prompt_key for item in self.prompts}) != len(self.prompts):
            raise PromptCoverageError("INTENT prompt keys must be unique")
        cluster_ids = [item.cluster_id for item in self.clusters]
        if len(set(cluster_ids)) != len(cluster_ids):
            raise PromptCoverageError("Offer cluster IDs must be unique")
        clustered_offer_ids = [offer_id for item in self.clusters for offer_id in item.offer_ids]
        if len(set(clustered_offer_ids)) != len(clustered_offer_ids):
            raise PromptCoverageError("Offer clusters contain duplicate offer IDs")
        known_clusters = set(cluster_ids)
        support: dict[str, list[str]] = {cluster_id: [] for cluster_id in cluster_ids}
        for prompt in self.prompts:
            unknown = set(prompt.supporting_cluster_ids).difference(known_clusters)
            if unknown:
                raise PromptCoverageError("Prompt references unknown offer clusters")
            for cluster_id in prompt.supporting_cluster_ids:
                support[cluster_id].append(prompt.prompt_key)
        exclusion_map = {item.cluster_id: item.reason for item in self.exclusions}
        if len(exclusion_map) != len(self.exclusions):
            raise PromptCoverageError("A cluster may have at most one exclusion")
        if set(exclusion_map).difference(known_clusters):
            raise PromptCoverageError("Exclusion references unknown offer cluster")
        expected_coverage: list[dict[str, Any]] = []
        for cluster in self.clusters:
            prompt_keys = sorted(support[cluster.cluster_id])
            reason = exclusion_map.get(cluster.cluster_id)
            if bool(prompt_keys) == bool(reason):
                raise PromptCoverageError(
                    "Each offer cluster needs prompts or one exclusion, but not both"
                )
            expected_coverage.append(
                {
                    "cluster_id": cluster.cluster_id,
                    "offer_ids": list(cluster.offer_ids),
                    "user_jobs": list(cluster.user_jobs),
                    "supporting_prompt_keys": prompt_keys,
                    "exclusion_reason": reason,
                }
            )
        if _canonical_json(expected_coverage) != _canonical_json(list(self.coverage)):
            raise PromptCoverageError("Coverage rows disagree with prompts and exclusions")


def build_prompt_foundation(
    *,
    upstream: UpstreamArtifactDigests,
    catalog: OfferCatalog,
    clusters: Sequence[OfferCluster],
    prompts: Sequence[IntentPrompt | Mapping[str, Any]],
    exclusions: Sequence[ClusterExclusion] = (),
    profile: Mapping[str, Any] | None = None,
    source_units: Iterable[SourceUnit | Mapping[str, Any]] = (),
    source_absence_claims_allowed: bool = False,
    crawl_coverage_state: str = "unknown",
) -> PromptFoundation:
    """Bind six INTENT prompts to every accepted offer cluster or exclusion."""

    catalog.validate()
    if upstream.catalog_digest != catalog.catalog_digest:
        raise PromptCoverageError("Upstream catalog digest does not match catalog")
    if not catalog.accepted_offers:
        if profile is None:
            raise OfferCatalogAdmissionError(
                "An empty offer catalog requires a grounded site profile admission"
            )
        admission = assess_offer_catalog_admission(
            catalog=catalog,
            profile=profile,
            source_units=source_units,
            source_absence_claims_allowed=source_absence_claims_allowed,
            crawl_coverage_state=crawl_coverage_state,
        )
        if admission.profile_digest != upstream.profile_digest:
            raise OfferCatalogAdmissionError(
                "Zero-offer admission profile digest does not match upstream"
            )
        admission.require_allowed()
    parsed_prompts = tuple(
        item if isinstance(item, IntentPrompt) else IntentPrompt.from_mapping(item)
        for item in prompts
    )
    if len(parsed_prompts) != len(INTENT_CODES):
        raise PromptCoverageError("Prompt foundation requires exactly six INTENT prompts")
    if {item.intent for item in parsed_prompts} != set(INTENT_CODES):
        raise PromptCoverageError("Prompt foundation requires each INTENT code exactly once")
    if len({item.prompt_key for item in parsed_prompts}) != len(parsed_prompts):
        raise PromptCoverageError("INTENT prompt keys must be unique")

    cluster_tuple = tuple(sorted(clusters, key=lambda item: item.cluster_id))
    cluster_ids = {item.cluster_id for item in cluster_tuple}
    catalog_offer_ids = {item.offer_id for item in catalog.accepted_offers}
    clustered_offer_ids = [offer_id for item in cluster_tuple for offer_id in item.offer_ids]
    if set(clustered_offer_ids) != catalog_offer_ids or len(clustered_offer_ids) != len(
        catalog_offer_ids
    ):
        raise PromptCoverageError("Offer clusters must partition every accepted offer exactly once")

    exclusion_tuple = tuple(sorted(exclusions, key=lambda item: item.cluster_id))
    exclusion_map = {item.cluster_id: item.reason for item in exclusion_tuple}
    if len(exclusion_map) != len(exclusion_tuple):
        raise PromptCoverageError("A cluster may have at most one exclusion")
    unknown_exclusions = set(exclusion_map).difference(cluster_ids)
    if unknown_exclusions:
        raise PromptCoverageError(f"Exclusions reference unknown clusters: {sorted(unknown_exclusions)}")

    support: dict[str, list[str]] = {cluster_id: [] for cluster_id in cluster_ids}
    for prompt in parsed_prompts:
        unknown = set(prompt.supporting_cluster_ids).difference(cluster_ids)
        if unknown:
            raise PromptCoverageError(
                f"Prompt {prompt.prompt_key!r} references unknown clusters: {sorted(unknown)}"
            )
        for cluster_id in prompt.supporting_cluster_ids:
            support[cluster_id].append(prompt.prompt_key)

    coverage: list[dict[str, Any]] = []
    for cluster in cluster_tuple:
        prompt_keys = sorted(support[cluster.cluster_id])
        exclusion_reason = exclusion_map.get(cluster.cluster_id)
        if prompt_keys and exclusion_reason is not None:
            raise PromptCoverageError(
                f"Cluster {cluster.cluster_id} is both covered and excluded"
            )
        if not prompt_keys and exclusion_reason is None:
            raise PromptCoverageError(
                f"Cluster {cluster.cluster_id} has no supporting prompt or exclusion"
            )
        coverage.append(
            {
                "cluster_id": cluster.cluster_id,
                "offer_ids": list(cluster.offer_ids),
                "user_jobs": list(cluster.user_jobs),
                "supporting_prompt_keys": prompt_keys,
                "exclusion_reason": exclusion_reason,
            }
        )

    ordered_prompts = tuple(sorted(parsed_prompts, key=lambda item: INTENT_CODES.index(item.intent)))
    prompt_set_digest = artifact_digest([item.as_dict() for item in ordered_prompts])
    coverage_tuple = tuple(coverage)
    coverage_digest = artifact_digest(list(coverage_tuple))
    provisional = PromptFoundation(
        upstream=upstream,
        clusters=cluster_tuple,
        prompts=ordered_prompts,
        exclusions=exclusion_tuple,
        coverage=coverage_tuple,
        prompt_set_digest=prompt_set_digest,
        coverage_digest=coverage_digest,
        foundation_digest="0" * 64,
    )
    foundation = PromptFoundation(
        upstream=upstream,
        clusters=cluster_tuple,
        prompts=ordered_prompts,
        exclusions=exclusion_tuple,
        coverage=coverage_tuple,
        prompt_set_digest=prompt_set_digest,
        coverage_digest=coverage_digest,
        foundation_digest=artifact_digest(provisional._body()),
    )
    foundation.validate()
    return foundation


@dataclass(frozen=True, slots=True)
class AnswerSetReceipt:
    prompt_foundation_digest: str
    prompt_set_digest: str
    prompt_keys: tuple[str, ...]
    answer_set_digest: str
    version: str = ANSWER_SET_RECEIPT_VERSION

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AnswerSetReceipt":
        version = str(value.get("version", ""))
        if version != ANSWER_SET_RECEIPT_VERSION:
            raise ResumeCompatibilityError("Unsupported answer set receipt version")
        prompt_keys = value.get("prompt_keys", ())
        if isinstance(prompt_keys, str) or not isinstance(prompt_keys, Sequence):
            raise ResumeCompatibilityError("Answer receipt prompt_keys must be an array")
        receipt = cls(
            prompt_foundation_digest=str(value.get("prompt_foundation_digest", "")),
            prompt_set_digest=str(value.get("prompt_set_digest", "")),
            prompt_keys=tuple(str(item) for item in prompt_keys),
            answer_set_digest=str(value.get("answer_set_digest", "")),
            version=version,
        )
        receipt.validate()
        return receipt

    def validate(self) -> None:
        _require_sha256(
            self.prompt_foundation_digest,
            field="prompt_foundation_digest",
        )
        _require_sha256(self.prompt_set_digest, field="prompt_set_digest")
        _require_sha256(self.answer_set_digest, field="answer_set_digest")
        if len(set(self.prompt_keys)) != len(self.prompt_keys):
            raise ResumeCompatibilityError("Answer receipt prompt keys must be unique")

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "prompt_foundation_digest": self.prompt_foundation_digest,
            "prompt_set_digest": self.prompt_set_digest,
            "prompt_keys": list(self.prompt_keys),
            "answer_set_digest": self.answer_set_digest,
        }


def build_answer_set_receipt(
    foundation: PromptFoundation,
    answers_by_prompt: Mapping[str, Any],
) -> AnswerSetReceipt:
    foundation.validate()
    expected = set(foundation.prompt_keys)
    actual = set(answers_by_prompt)
    if actual != expected:
        missing = sorted(expected.difference(actual))
        extra = sorted(actual.difference(expected))
        raise ResumeCompatibilityError(
            f"Answers do not match prompt foundation; missing={missing}, extra={extra}"
        )
    ordered_answers = {
        key: answers_by_prompt[key] for key in sorted(answers_by_prompt)
    }
    receipt = AnswerSetReceipt(
        prompt_foundation_digest=foundation.foundation_digest,
        prompt_set_digest=foundation.prompt_set_digest,
        prompt_keys=tuple(sorted(expected)),
        answer_set_digest=artifact_digest(ordered_answers),
    )
    receipt.validate()
    return receipt


@dataclass(frozen=True, slots=True)
class ResumeCompatibilityReport:
    compatible: bool
    mismatches: tuple[str, ...]


def audit_resume_compatibility(
    *,
    current_upstream: UpstreamArtifactDigests | Mapping[str, Any],
    persisted_foundation: PromptFoundation | Mapping[str, Any],
    answer_receipt: AnswerSetReceipt | Mapping[str, Any] | None = None,
    persisted_answers_by_prompt: Mapping[str, Any] | None = None,
) -> ResumeCompatibilityReport:
    """Audit, without mutating, whether persisted downstream work is reusable."""

    if not isinstance(current_upstream, UpstreamArtifactDigests):
        current_upstream = UpstreamArtifactDigests.from_mapping(current_upstream)
    if not isinstance(persisted_foundation, PromptFoundation):
        try:
            persisted_foundation = PromptFoundation.from_mapping(persisted_foundation)
        except OfferCatalogError as exc:
            return ResumeCompatibilityReport(
                compatible=False,
                mismatches=(f"invalid_prompt_foundation:{exc}",),
            )
    if answer_receipt is not None and not isinstance(answer_receipt, AnswerSetReceipt):
        try:
            answer_receipt = AnswerSetReceipt.from_mapping(answer_receipt)
        except OfferCatalogError as exc:
            return ResumeCompatibilityReport(
                compatible=False,
                mismatches=(f"invalid_answer_receipt:{exc}",),
            )

    mismatches: list[str] = []
    try:
        persisted_foundation.validate()
    except OfferCatalogError as exc:
        mismatches.append(f"invalid_prompt_foundation:{exc}")
    for field in (
        "profile_digest",
        "catalog_digest",
        "market_research_digest",
        "selected_pages_digest",
    ):
        current_value = getattr(current_upstream, field)
        persisted_value = getattr(persisted_foundation.upstream, field)
        if current_value != persisted_value:
            mismatches.append(f"{field}:current={current_value}:persisted={persisted_value}")

    if answer_receipt is not None:
        try:
            answer_receipt.validate()
        except OfferCatalogError as exc:
            mismatches.append(f"invalid_answer_receipt:{exc}")
        if answer_receipt.prompt_foundation_digest != persisted_foundation.foundation_digest:
            mismatches.append("answer_prompt_foundation_digest_mismatch")
        if answer_receipt.prompt_set_digest != persisted_foundation.prompt_set_digest:
            mismatches.append("answer_prompt_set_digest_mismatch")
        if set(answer_receipt.prompt_keys) != set(persisted_foundation.prompt_keys):
            mismatches.append("answer_prompt_keys_mismatch")
        try:
            _require_sha256(answer_receipt.answer_set_digest, field="answer_set_digest")
        except OfferCatalogError as exc:
            mismatches.append(f"invalid_answer_set_digest:{exc}")
        if persisted_answers_by_prompt is not None:
            expected_keys = set(persisted_foundation.prompt_keys)
            if set(persisted_answers_by_prompt) != expected_keys:
                mismatches.append("persisted_answer_keys_mismatch")
            else:
                actual_digest = artifact_digest(
                    {
                        key: persisted_answers_by_prompt[key]
                        for key in sorted(persisted_answers_by_prompt)
                    }
                )
                if actual_digest != answer_receipt.answer_set_digest:
                    mismatches.append("answer_set_content_digest_mismatch")
    elif persisted_answers_by_prompt is not None:
        mismatches.append("persisted_answers_have_no_answer_receipt")

    return ResumeCompatibilityReport(
        compatible=not mismatches,
        mismatches=tuple(mismatches),
    )


def validate_resume_compatibility(
    *,
    current_upstream: UpstreamArtifactDigests | Mapping[str, Any],
    persisted_foundation: PromptFoundation | Mapping[str, Any],
    answer_receipt: AnswerSetReceipt | Mapping[str, Any] | None = None,
    persisted_answers_by_prompt: Mapping[str, Any] | None = None,
) -> ResumeCompatibilityReport:
    """Fail closed when a resume would mix different analysis foundations."""

    report = audit_resume_compatibility(
        current_upstream=current_upstream,
        persisted_foundation=persisted_foundation,
        answer_receipt=answer_receipt,
        persisted_answers_by_prompt=persisted_answers_by_prompt,
    )
    if not report.compatible:
        raise ResumeCompatibilityError("; ".join(report.mismatches))
    return report


__all__ = [
    "ANSWER_SET_RECEIPT_VERSION",
    "CLIENT_IDENTITY_ADMISSION_VERSION",
    "DOMAIN_RESEARCH_MANIFEST_VERSION",
    "DOMAIN_RESEARCH_PAYLOAD_VERSION",
    "INTENT_CODES",
    "MAX_ACCEPTED_OFFERS",
    "OFFER_CATALOG_VERSION",
    "OFFER_USER_JOB_EVIDENCE_VERSION",
    "AcceptedOffer",
    "AnswerSetReceipt",
    "ClusterExclusion",
    "DispositionDecision",
    "DomainResearchPayload",
    "IntentPrompt",
    "OfferCandidate",
    "OfferCatalog",
    "OfferCatalogError",
    "OfferCluster",
    "OfferDisposition",
    "OfferEvidenceError",
    "OfferEvidenceRef",
    "OfferUserJobEvidence",
    "OfferKind",
    "PayloadShard",
    "PromptCoverageError",
    "PromptFoundation",
    "ResumeCompatibilityError",
    "ResumeCompatibilityReport",
    "SourceUnit",
    "UpstreamArtifactDigests",
    "artifact_digest",
    "admit_client_identity_aliases",
    "audit_resume_compatibility",
    "build_answer_set_receipt",
    "build_domain_research_payload",
    "build_offer_catalog",
    "build_offer_clusters",
    "build_prompt_foundation",
    "build_upstream_artifact_digests",
    "reconstruct_domain_research_payload",
    "validate_resume_compatibility",
]
