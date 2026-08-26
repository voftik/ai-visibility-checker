from __future__ import annotations

"""Content-addressed arbitration for standalone offer identities.

The offer catalog proves that the analyzed client offers a product, service,
or direction.  It does not, by itself, prove that the offer's bare name is a
distinctive product identity.  This module keeps those two claims separate.

One primary classifier and one independent critic receive the same immutable,
catalog-wide evidence packet.  Each role classifies every canonical name and
alias in one aggregate response.  A name is eligible for standalone matching
only when both valid responses independently classify it as ``named_offering``.
Malformed, missing, duplicated, stale, or disagreeing responses resolve to
``ambiguous``.  Static generic-category vocabulary is a hard code-owned
override.  Model failures therefore reduce metric recall without stopping the
report or manufacturing visibility.

The module is deliberately transport-agnostic.  It emits exactly one bounded
aggregate request per role and the corresponding JSON schema; the caller owns
the two provider executions and durable receipts.
"""

import copy
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from app.services.offer_catalog import (
    MAX_ACCEPTED_OFFERS,
    AcceptedOffer,
    OfferCatalog,
    OfferCatalogError,
    artifact_digest,
    is_generic_offer_name,
)

OFFER_IDENTITY_POLICY_VERSION = "aiv-offer-identity-policy-v1"
OFFER_IDENTITY_CONTRACT_VERSION = "aiv-offer-identity-contract-v1"
OFFER_IDENTITY_SUBJECT_VERSION = "aiv-offer-identity-subject-v1"
OFFER_IDENTITY_MODEL_DECISION_VERSION = "aiv-offer-identity-model-decision-v1"
OFFER_IDENTITY_MODEL_BATCH_VERSION = "aiv-offer-identity-model-batch-v1"
OFFER_IDENTITY_REQUEST_CONTRACT_VERSION = "aiv-offer-identity-request-v1"
OFFER_IDENTITY_RESULT_VERSION = "aiv-offer-identity-result-v1"
OFFER_IDENTITY_PRIMARY_PROMPT_VERSION = "aiv-offer-identity-primary-v1"
OFFER_IDENTITY_CRITIC_PROMPT_VERSION = "aiv-offer-identity-critic-v1"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MODEL_DECISION_KEYS = frozenset(
    {
        "version",
        "role",
        "subject_id",
        "input_digest",
        "policy_digest",
        "catalog_digest",
        "subject_digest",
        "evidence_refs_digest",
        "reviewed_evidence_ref_digests",
        "decision",
        "reason_code",
        "rationale",
    }
)
_MODEL_BATCH_KEYS = frozenset(
    {
        "version",
        "role",
        "input_digest",
        "policy_digest",
        "catalog_digest",
        "request_contract_digest",
        "subject_count",
        "decisions",
    }
)
_MODEL_RECEIPT_KEYS = frozenset(
    {
        "role",
        "subject_id",
        "valid",
        "decision",
        "reason_code",
        "rationale",
        "input_digest",
        "policy_digest",
        "catalog_digest",
        "subject_digest",
        "evidence_refs_digest",
        "reviewed_evidence_ref_digests",
        "receipt_digest",
    }
)
_RESOLVED_DECISION_KEYS = frozenset(
    {
        "subject_id",
        "offer_id",
        "name",
        "name_role",
        "decision",
        "standalone",
        "resolution_code",
        "primary",
        "critic",
        "decision_digest",
    }
)
_RESULT_KEYS = frozenset(
    {
        "version",
        "input_digest",
        "policy_digest",
        "catalog_digest",
        "decision_count",
        "standalone_count",
        "ambiguous_count",
        "decisions",
        "diagnostics",
        "output_digest",
    }
)


class OfferIdentityPolicyError(ValueError):
    """An immutable policy input is malformed or no longer catalog-bound."""


class OfferIdentityDecision(str, Enum):
    NAMED_OFFERING = "named_offering"
    GENERIC_CATEGORY = "generic_category"
    AMBIGUOUS = "ambiguous"


class OfferIdentityModelRole(str, Enum):
    PRIMARY = "primary"
    CRITIC = "critic"


class OfferIdentityNameRole(str, Enum):
    CANONICAL = "canonical"
    ALIAS = "alias"


class OfferIdentityReasonCode(str, Enum):
    EXPLICIT_NAMED_IDENTITY = "explicit_named_identity"
    DESCRIPTIVE_OFFERING_PHRASE = "descriptive_offering_phrase"
    INSUFFICIENT_IDENTITY_EVIDENCE = "insufficient_identity_evidence"
    CONFLICTING_IDENTITY_EVIDENCE = "conflicting_identity_evidence"


_REASONS_BY_DECISION = {
    OfferIdentityDecision.NAMED_OFFERING: frozenset(
        {OfferIdentityReasonCode.EXPLICIT_NAMED_IDENTITY}
    ),
    OfferIdentityDecision.GENERIC_CATEGORY: frozenset(
        {OfferIdentityReasonCode.DESCRIPTIVE_OFFERING_PHRASE}
    ),
    OfferIdentityDecision.AMBIGUOUS: frozenset(
        {
            OfferIdentityReasonCode.INSUFFICIENT_IDENTITY_EVIDENCE,
            OfferIdentityReasonCode.CONFLICTING_IDENTITY_EVIDENCE,
        }
    ),
}


OFFER_IDENTITY_POLICY: dict[str, Any] = {
    "version": OFFER_IDENTITY_POLICY_VERSION,
    "decisions": [item.value for item in OfferIdentityDecision],
    "model_roles": [item.value for item in OfferIdentityModelRole],
    "unit_of_work": "one_complete_catalog_per_role_with_all_exact_offer_evidence",
    "independence": "critic_does_not_receive_primary_output",
    "standalone_admission": (
        "primary_and_critic_are_valid_and_both_equal_named_offering"
    ),
    "static_override": "is_generic_offer_name_forces_generic_category",
    "failure_policy": "missing_invalid_duplicate_extra_or_disagreement_is_ambiguous",
    "raw_data_policy": "derived_policy_only_raw_answers_unchanged",
}
OFFER_IDENTITY_POLICY_DIGEST = artifact_digest(OFFER_IDENTITY_POLICY)


OFFER_IDENTITY_PRIMARY_SYSTEM_PROMPT = """
Ты классифицируешь все имена из доказанного каталога предложений клиента.
Каталог уже подтвердил, что клиент предлагает соответствующий товар или
услугу. Это не доказывает, что имя можно считать самостоятельным собственным
названием.

Верни named_offering, только если точные фрагменты источника показывают, что
имя используется как отличимое название конкретного продукта, платформы,
сервиса или линейки. Верни generic_category, если это обычное название товара,
услуги, рынка, технологии или возможности. Верни ambiguous, если буквальных
доказательств недостаточно или они противоречат друг другу.

Классифицируй каждый subject из пакета ровно один раз. Не используй знания вне
переданных evidence_refs. Не подменяй проверку
принадлежности проверкой имени: фраза «мы продаём X» доказывает предложение X,
но сама по себе не делает X собственным названием. Проверь все evidence_refs и
скопируй их digests в исходном порядке. Не пропускай и не добавляй subject’ы.
Ответь строго по JSON-схеме.
""".strip()


OFFER_IDENTITY_CRITIC_SYSTEM_PROMPT = """
Ты независимый критик классификации всех имён из каталога предложений.
Первичный вывод тебе не передаётся. Проведи собственную проверку каждого
subject из пакета только по точным evidence_refs.

named_offering допустим лишь для отличимого собственного названия конкретного
продукта, платформы, сервиса или линейки. Обычное название товара, услуги,
рынка, технологии или возможности — generic_category. Если источника не
хватает для уверенного различения, выбери ambiguous.

Принадлежность предложения клиенту уже доказана каталогом, но не означает,
что голая фраза однозначно идентифицирует его продукт. Проверь все
evidence_refs, верни их digests в исходном порядке, не пропускай и не добавляй
subject’ы и ответь строго по JSON-схеме.
""".strip()


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise OfferIdentityPolicyError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: Iterable[str],
    *,
    field: str,
) -> None:
    expected_keys = frozenset(expected)
    actual_keys = frozenset(value)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise OfferIdentityPolicyError(
            f"{field} keys mismatch; missing={missing}, extra={extra}"
        )


def _identity_key(value: str) -> str:
    return " ".join(str(value or "").casefold().replace("ё", "е").split())


def _evidence_mappings(offer: AcceptedOffer) -> tuple[dict[str, Any], ...]:
    return tuple(copy.deepcopy(item.as_dict()) for item in offer.evidence_refs)


@dataclass(frozen=True, slots=True)
class OfferIdentitySubject:
    subject_id: str
    ordinal: int
    offer_id: str
    offer_kind: str
    name_role: OfferIdentityNameRole
    name: str
    offer_digest: str
    evidence_refs: tuple[dict[str, Any], ...]
    evidence_ref_digests: tuple[str, ...]
    evidence_refs_digest: str
    subject_digest: str
    version: str = OFFER_IDENTITY_SUBJECT_VERSION

    def _body(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "subject_id": self.subject_id,
            "ordinal": self.ordinal,
            "offer_id": self.offer_id,
            "offer_kind": self.offer_kind,
            "name_role": self.name_role.value,
            "name": self.name,
            "offer_digest": self.offer_digest,
            "evidence_refs": [copy.deepcopy(item) for item in self.evidence_refs],
            "evidence_ref_digests": list(self.evidence_ref_digests),
            "evidence_refs_digest": self.evidence_refs_digest,
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self._body(), "subject_digest": self.subject_digest}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> OfferIdentitySubject:
        _require_exact_keys(
            value,
            {
                "version",
                "subject_id",
                "ordinal",
                "offer_id",
                "offer_kind",
                "name_role",
                "name",
                "offer_digest",
                "evidence_refs",
                "evidence_ref_digests",
                "evidence_refs_digest",
                "subject_digest",
            },
            field="subject",
        )
        raw_refs = value.get("evidence_refs")
        raw_ref_digests = value.get("evidence_ref_digests")
        if (
            isinstance(raw_refs, (str, bytes))
            or not isinstance(raw_refs, Sequence)
            or isinstance(raw_ref_digests, (str, bytes))
            or not isinstance(raw_ref_digests, Sequence)
        ):
            raise OfferIdentityPolicyError("subject evidence arrays are malformed")
        ordinal = value.get("ordinal")
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
            raise OfferIdentityPolicyError("subject ordinal is invalid")
        try:
            role = OfferIdentityNameRole(str(value.get("name_role") or ""))
        except ValueError as exc:
            raise OfferIdentityPolicyError("subject name_role is invalid") from exc
        subject = cls(
            version=str(value.get("version") or ""),
            subject_id=str(value.get("subject_id") or ""),
            ordinal=ordinal,
            offer_id=str(value.get("offer_id") or ""),
            offer_kind=str(value.get("offer_kind") or ""),
            name_role=role,
            name=str(value.get("name") or ""),
            offer_digest=_require_sha256(value.get("offer_digest"), field="offer_digest"),
            evidence_refs=tuple(copy.deepcopy(item) for item in raw_refs),
            evidence_ref_digests=tuple(
                _require_sha256(item, field="evidence_ref_digest")
                for item in raw_ref_digests
            ),
            evidence_refs_digest=_require_sha256(
                value.get("evidence_refs_digest"), field="evidence_refs_digest"
            ),
            subject_digest=_require_sha256(
                value.get("subject_digest"), field="subject_digest"
            ),
        )
        subject.validate()
        return subject

    def validate(self) -> None:
        if self.version != OFFER_IDENTITY_SUBJECT_VERSION:
            raise OfferIdentityPolicyError("unsupported offer identity subject version")
        if not self.subject_id.startswith("identity:") or not self.offer_id or not self.name:
            raise OfferIdentityPolicyError("subject identity is incomplete")
        if not self.evidence_refs:
            raise OfferIdentityPolicyError("subject has no exact evidence refs")
        expected_ref_digests = tuple(artifact_digest(item) for item in self.evidence_refs)
        if self.evidence_ref_digests != expected_ref_digests:
            raise OfferIdentityPolicyError("subject evidence ref digests mismatch")
        if self.evidence_refs_digest != artifact_digest(list(self.evidence_refs)):
            raise OfferIdentityPolicyError("subject evidence bundle digest mismatch")
        if self.subject_digest != artifact_digest(self._body()):
            raise OfferIdentityPolicyError("subject digest mismatch")
        expected_id = "identity:" + artifact_digest(
            {
                "offer_id": self.offer_id,
                "name_role": self.name_role.value,
                "name": self.name,
                "ordinal": self.ordinal,
                "offer_digest": self.offer_digest,
                "evidence_refs_digest": self.evidence_refs_digest,
            }
        )
        if self.subject_id != expected_id:
            raise OfferIdentityPolicyError("subject_id content identity mismatch")


def _subject_from_offer(
    offer: AcceptedOffer,
    *,
    name: str,
    name_role: OfferIdentityNameRole,
    ordinal: int,
) -> OfferIdentitySubject:
    offer_digest = artifact_digest(offer.as_dict())
    evidence_refs = _evidence_mappings(offer)
    evidence_ref_digests = tuple(artifact_digest(item) for item in evidence_refs)
    evidence_refs_digest = artifact_digest(list(evidence_refs))
    subject_id = "identity:" + artifact_digest(
        {
            "offer_id": offer.offer_id,
            "name_role": name_role.value,
            "name": name,
            "ordinal": ordinal,
            "offer_digest": offer_digest,
            "evidence_refs_digest": evidence_refs_digest,
        }
    )
    body = {
        "version": OFFER_IDENTITY_SUBJECT_VERSION,
        "subject_id": subject_id,
        "ordinal": ordinal,
        "offer_id": offer.offer_id,
        "offer_kind": offer.kind.value,
        "name_role": name_role.value,
        "name": name,
        "offer_digest": offer_digest,
        "evidence_refs": [copy.deepcopy(item) for item in evidence_refs],
        "evidence_ref_digests": list(evidence_ref_digests),
        "evidence_refs_digest": evidence_refs_digest,
    }
    subject = OfferIdentitySubject(
        subject_id=subject_id,
        ordinal=ordinal,
        offer_id=offer.offer_id,
        offer_kind=offer.kind.value,
        name_role=name_role,
        name=name,
        offer_digest=offer_digest,
        evidence_refs=evidence_refs,
        evidence_ref_digests=evidence_ref_digests,
        evidence_refs_digest=evidence_refs_digest,
        subject_digest=artifact_digest(body),
    )
    subject.validate()
    return subject


@dataclass(frozen=True, slots=True)
class OfferIdentityContract:
    catalog_digest: str
    catalog_artifact_digest: str
    catalog_evidence_digest: str
    subjects: tuple[OfferIdentitySubject, ...]
    input_digest: str
    policy_digest: str = OFFER_IDENTITY_POLICY_DIGEST
    policy_version: str = OFFER_IDENTITY_POLICY_VERSION
    version: str = OFFER_IDENTITY_CONTRACT_VERSION

    def _body(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "policy_version": self.policy_version,
            "policy_digest": self.policy_digest,
            "catalog_digest": self.catalog_digest,
            "catalog_artifact_digest": self.catalog_artifact_digest,
            "catalog_evidence_digest": self.catalog_evidence_digest,
            "subject_count": len(self.subjects),
            "subjects": [item.as_dict() for item in self.subjects],
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self._body(), "input_digest": self.input_digest}

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        catalog: OfferCatalog,
    ) -> OfferIdentityContract:
        _require_exact_keys(
            value,
            {
                "version",
                "policy_version",
                "policy_digest",
                "catalog_digest",
                "catalog_artifact_digest",
                "catalog_evidence_digest",
                "subject_count",
                "subjects",
                "input_digest",
            },
            field="contract",
        )
        raw_subjects = value.get("subjects")
        if isinstance(raw_subjects, (str, bytes)) or not isinstance(
            raw_subjects, Sequence
        ):
            raise OfferIdentityPolicyError("contract subjects are malformed")
        subject_count = value.get("subject_count")
        if (
            isinstance(subject_count, bool)
            or not isinstance(subject_count, int)
            or subject_count != len(raw_subjects)
        ):
            raise OfferIdentityPolicyError("contract subject_count mismatch")
        contract = cls(
            version=str(value.get("version") or ""),
            policy_version=str(value.get("policy_version") or ""),
            policy_digest=_require_sha256(
                value.get("policy_digest"), field="policy_digest"
            ),
            catalog_digest=_require_sha256(
                value.get("catalog_digest"), field="catalog_digest"
            ),
            catalog_artifact_digest=_require_sha256(
                value.get("catalog_artifact_digest"),
                field="catalog_artifact_digest",
            ),
            catalog_evidence_digest=_require_sha256(
                value.get("catalog_evidence_digest"),
                field="catalog_evidence_digest",
            ),
            subjects=tuple(
                OfferIdentitySubject.from_mapping(item) for item in raw_subjects
            ),
            input_digest=_require_sha256(
                value.get("input_digest"), field="input_digest"
            ),
        )
        contract.validate_against(catalog)
        return contract

    def validate_against(self, catalog: OfferCatalog) -> None:
        try:
            catalog.validate()
        except OfferCatalogError as exc:
            raise OfferIdentityPolicyError("offer catalog is invalid") from exc
        if self.version != OFFER_IDENTITY_CONTRACT_VERSION:
            raise OfferIdentityPolicyError("unsupported offer identity contract version")
        if (
            self.policy_version != OFFER_IDENTITY_POLICY_VERSION
            or self.policy_digest != OFFER_IDENTITY_POLICY_DIGEST
        ):
            raise OfferIdentityPolicyError("offer identity policy binding mismatch")
        if self.catalog_digest != catalog.catalog_digest:
            raise OfferIdentityPolicyError("contract catalog_digest mismatch")
        if self.catalog_artifact_digest != artifact_digest(catalog.as_dict()):
            raise OfferIdentityPolicyError("contract catalog artifact digest mismatch")
        expected = build_offer_identity_contract(catalog, _validate_catalog=False)
        if self.catalog_evidence_digest != expected.catalog_evidence_digest:
            raise OfferIdentityPolicyError("contract evidence manifest mismatch")
        if len(self.subjects) != len(expected.subjects):
            raise OfferIdentityPolicyError("contract subject coverage mismatch")
        for ordinal, subject in enumerate(self.subjects):
            subject.validate()
            if subject.ordinal != ordinal:
                raise OfferIdentityPolicyError("contract subject order is not contiguous")
        subject_ids = [item.subject_id for item in self.subjects]
        if len(subject_ids) != len(set(subject_ids)):
            raise OfferIdentityPolicyError("contract contains duplicate subjects")
        if [item.as_dict() for item in self.subjects] != [
            item.as_dict() for item in expected.subjects
        ]:
            raise OfferIdentityPolicyError(
                "contract names or exact evidence differ from the catalog"
            )
        if self.input_digest != artifact_digest(self._body()):
            raise OfferIdentityPolicyError("contract input_digest mismatch")


def build_offer_identity_contract(
    catalog: OfferCatalog,
    *,
    _validate_catalog: bool = True,
) -> OfferIdentityContract:
    if _validate_catalog:
        try:
            catalog.validate()
        except OfferCatalogError as exc:
            raise OfferIdentityPolicyError("offer catalog is invalid") from exc
    if len(catalog.accepted_offers) > MAX_ACCEPTED_OFFERS:
        raise OfferIdentityPolicyError("offer catalog exceeds the 10-offer contract")
    subjects: list[OfferIdentitySubject] = []
    occurrence_keys: set[tuple[str, str, str]] = set()
    for offer in catalog.accepted_offers:
        names = (
            (OfferIdentityNameRole.CANONICAL, offer.canonical_name),
            *(
                (OfferIdentityNameRole.ALIAS, alias)
                for alias in offer.aliases
            ),
        )
        for name_role, name in names:
            occurrence_key = (offer.offer_id, name_role.value, _identity_key(name))
            if occurrence_key in occurrence_keys:
                raise OfferIdentityPolicyError(
                    "catalog repeats a canonical or alias occurrence"
                )
            occurrence_keys.add(occurrence_key)
            subjects.append(
                _subject_from_offer(
                    offer,
                    name=name,
                    name_role=name_role,
                    ordinal=len(subjects),
                )
            )
    evidence_manifest = [
        {
            "offer_id": offer.offer_id,
            "offer_digest": artifact_digest(offer.as_dict()),
            "evidence_refs": [item.as_dict() for item in offer.evidence_refs],
            "evidence_ref_digests": [
                artifact_digest(item.as_dict()) for item in offer.evidence_refs
            ],
        }
        for offer in catalog.accepted_offers
    ]
    partial = OfferIdentityContract(
        catalog_digest=catalog.catalog_digest,
        catalog_artifact_digest=artifact_digest(catalog.as_dict()),
        catalog_evidence_digest=artifact_digest(evidence_manifest),
        subjects=tuple(subjects),
        input_digest="0" * 64,
    )
    return OfferIdentityContract(
        catalog_digest=partial.catalog_digest,
        catalog_artifact_digest=partial.catalog_artifact_digest,
        catalog_evidence_digest=partial.catalog_evidence_digest,
        subjects=partial.subjects,
        input_digest=artifact_digest(partial._body()),
    )


def _subject_decision_schema(role: OfferIdentityModelRole) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "version": {"type": "string", "const": OFFER_IDENTITY_MODEL_DECISION_VERSION},
            "role": {"type": "string", "const": role.value},
            "subject_id": {"type": "string", "pattern": r"^identity:[0-9a-f]{64}$"},
            "input_digest": {"type": "string", "pattern": r"^[0-9a-f]{64}$"},
            "policy_digest": {"type": "string", "pattern": r"^[0-9a-f]{64}$"},
            "catalog_digest": {"type": "string", "pattern": r"^[0-9a-f]{64}$"},
            "subject_digest": {"type": "string", "pattern": r"^[0-9a-f]{64}$"},
            "evidence_refs_digest": {"type": "string", "pattern": r"^[0-9a-f]{64}$"},
            "reviewed_evidence_ref_digests": {
                "type": "array",
                "items": {"type": "string", "pattern": r"^[0-9a-f]{64}$"},
                "uniqueItems": True,
            },
            "decision": {
                "type": "string",
                "enum": [item.value for item in OfferIdentityDecision],
            },
            "reason_code": {
                "type": "string",
                "enum": [item.value for item in OfferIdentityReasonCode],
            },
            "rationale": {"type": "string", "minLength": 1, "maxLength": 600},
        },
        "required": sorted(_MODEL_DECISION_KEYS),
    }


def _role_request_contract(
    contract: OfferIdentityContract,
    role: OfferIdentityModelRole,
) -> dict[str, Any]:
    prompt_version = (
        OFFER_IDENTITY_PRIMARY_PROMPT_VERSION
        if role is OfferIdentityModelRole.PRIMARY
        else OFFER_IDENTITY_CRITIC_PROMPT_VERSION
    )
    system_prompt = (
        OFFER_IDENTITY_PRIMARY_SYSTEM_PROMPT
        if role is OfferIdentityModelRole.PRIMARY
        else OFFER_IDENTITY_CRITIC_SYSTEM_PROMPT
    )
    body = {
        "version": OFFER_IDENTITY_REQUEST_CONTRACT_VERSION,
        "role": role.value,
        "prompt_version": prompt_version,
        "system_prompt_digest": artifact_digest({"text": system_prompt}),
        "response_version": OFFER_IDENTITY_MODEL_BATCH_VERSION,
        "input_digest": contract.input_digest,
        "policy_digest": contract.policy_digest,
        "catalog_digest": contract.catalog_digest,
        "subject_count": len(contract.subjects),
        "subject_digests": [item.subject_digest for item in contract.subjects],
        "model_parameters": {
            "temperature": 0.15,
            "web_policy": "forbidden",
            "response_mode": "structured_json",
        },
    }
    return {**body, "request_contract_digest": artifact_digest(body)}


def offer_identity_decision_schema(
    contract: OfferIdentityContract,
    role: OfferIdentityModelRole | str,
) -> dict[str, Any]:
    """Return the exact aggregate response schema for one independent role."""

    normalized_role = OfferIdentityModelRole(role)
    request_contract = _role_request_contract(contract, normalized_role)
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "version": {
                "type": "string",
                "const": OFFER_IDENTITY_MODEL_BATCH_VERSION,
            },
            "role": {"type": "string", "const": normalized_role.value},
            "input_digest": {"type": "string", "const": contract.input_digest},
            "policy_digest": {"type": "string", "const": contract.policy_digest},
            "catalog_digest": {"type": "string", "const": contract.catalog_digest},
            "request_contract_digest": {
                "type": "string",
                "const": request_contract["request_contract_digest"],
            },
            "subject_count": {
                "type": "integer",
                "const": len(contract.subjects),
            },
            "decisions": {
                "type": "array",
                "minItems": len(contract.subjects),
                "maxItems": len(contract.subjects),
                "items": _subject_decision_schema(normalized_role),
            },
        },
        "required": sorted(_MODEL_BATCH_KEYS),
    }


def build_offer_identity_model_request(
    contract: OfferIdentityContract,
    *,
    role: OfferIdentityModelRole | str,
) -> dict[str, Any]:
    """Return the one catalog-wide request for an independent model role."""

    normalized_role = OfferIdentityModelRole(role)
    request_contract = _role_request_contract(contract, normalized_role)
    payload = {
        "request_contract": request_contract,
        "contract": {
            "input_digest": contract.input_digest,
            "policy_digest": contract.policy_digest,
            "catalog_digest": contract.catalog_digest,
            "subject_count": len(contract.subjects),
        },
        "subjects": [item.as_dict() for item in contract.subjects],
        "decision_definitions": {
            "named_offering": "отличимое собственное название предложения",
            "generic_category": "обычное название товара, услуги, рынка или технологии",
            "ambiguous": "источника недостаточно для надёжного различения",
        },
    }
    prompt_version = (
        OFFER_IDENTITY_PRIMARY_PROMPT_VERSION
        if normalized_role is OfferIdentityModelRole.PRIMARY
        else OFFER_IDENTITY_CRITIC_PROMPT_VERSION
    )
    request_body = {
        "prompt_version": prompt_version,
        "role": normalized_role.value,
        "system_prompt": (
            OFFER_IDENTITY_PRIMARY_SYSTEM_PROMPT
            if normalized_role is OfferIdentityModelRole.PRIMARY
            else OFFER_IDENTITY_CRITIC_SYSTEM_PROMPT
        ),
        "payload": payload,
        "response_schema_name": f"offer_identity_{normalized_role.value}_v1",
        "response_schema": offer_identity_decision_schema(contract, normalized_role),
        "model_parameters": copy.deepcopy(request_contract["model_parameters"]),
    }
    return {**request_body, "request_digest": artifact_digest(request_body)}


@dataclass(frozen=True, slots=True)
class _ModelReceipt:
    role: OfferIdentityModelRole
    subject_id: str
    valid: bool
    decision: OfferIdentityDecision
    reason_code: str
    rationale: str
    input_digest: str
    policy_digest: str
    catalog_digest: str
    subject_digest: str
    evidence_refs_digest: str
    reviewed_evidence_ref_digests: tuple[str, ...]
    receipt_digest: str

    def _body(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "subject_id": self.subject_id,
            "valid": self.valid,
            "decision": self.decision.value,
            "reason_code": self.reason_code,
            "rationale": self.rationale,
            "input_digest": self.input_digest,
            "policy_digest": self.policy_digest,
            "catalog_digest": self.catalog_digest,
            "subject_digest": self.subject_digest,
            "evidence_refs_digest": self.evidence_refs_digest,
            "reviewed_evidence_ref_digests": list(
                self.reviewed_evidence_ref_digests
            ),
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self._body(), "receipt_digest": self.receipt_digest}

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        role: OfferIdentityModelRole,
        subject: OfferIdentitySubject,
        contract: OfferIdentityContract,
    ) -> _ModelReceipt:
        _require_exact_keys(value, _MODEL_RECEIPT_KEYS, field="model receipt")
        valid = value.get("valid")
        if not isinstance(valid, bool):
            raise OfferIdentityPolicyError("model receipt valid flag is invalid")
        try:
            decision = OfferIdentityDecision(str(value.get("decision") or ""))
        except ValueError as exc:
            raise OfferIdentityPolicyError("model receipt decision is invalid") from exc
        raw_reviewed = value.get("reviewed_evidence_ref_digests")
        if isinstance(raw_reviewed, (str, bytes)) or not isinstance(
            raw_reviewed, Sequence
        ):
            raise OfferIdentityPolicyError("model receipt evidence list is invalid")
        receipt = cls(
            role=OfferIdentityModelRole(str(value.get("role") or "")),
            subject_id=str(value.get("subject_id") or ""),
            valid=valid,
            decision=decision,
            reason_code=str(value.get("reason_code") or ""),
            rationale=str(value.get("rationale") or ""),
            input_digest=_require_sha256(
                value.get("input_digest"), field="receipt input_digest"
            ),
            policy_digest=_require_sha256(
                value.get("policy_digest"), field="receipt policy_digest"
            ),
            catalog_digest=_require_sha256(
                value.get("catalog_digest"), field="receipt catalog_digest"
            ),
            subject_digest=_require_sha256(
                value.get("subject_digest"), field="receipt subject_digest"
            ),
            evidence_refs_digest=_require_sha256(
                value.get("evidence_refs_digest"),
                field="receipt evidence_refs_digest",
            ),
            reviewed_evidence_ref_digests=tuple(
                _require_sha256(item, field="receipt evidence_ref_digest")
                for item in raw_reviewed
            ),
            receipt_digest=_require_sha256(
                value.get("receipt_digest"), field="receipt_digest"
            ),
        )
        bindings = {
            "role": role,
            "subject_id": subject.subject_id,
            "input_digest": contract.input_digest,
            "policy_digest": contract.policy_digest,
            "catalog_digest": contract.catalog_digest,
            "subject_digest": subject.subject_digest,
            "evidence_refs_digest": subject.evidence_refs_digest,
        }
        for field, expected in bindings.items():
            if getattr(receipt, field) != expected:
                raise OfferIdentityPolicyError(
                    f"model receipt {field} binding mismatch"
                )
        if receipt.valid:
            try:
                reason = OfferIdentityReasonCode(receipt.reason_code)
            except ValueError as exc:
                raise OfferIdentityPolicyError(
                    "valid model receipt reason is invalid"
                ) from exc
            if reason not in _REASONS_BY_DECISION[receipt.decision]:
                raise OfferIdentityPolicyError(
                    "valid model receipt reason contradicts decision"
                )
            if not 1 <= len(receipt.rationale) <= 600:
                raise OfferIdentityPolicyError(
                    "valid model receipt rationale is invalid"
                )
            if receipt.reviewed_evidence_ref_digests != subject.evidence_ref_digests:
                raise OfferIdentityPolicyError(
                    "valid model receipt evidence coverage mismatch"
                )
        elif (
            receipt.decision is not OfferIdentityDecision.AMBIGUOUS
            or receipt.rationale
            or receipt.reviewed_evidence_ref_digests
        ):
            raise OfferIdentityPolicyError("invalid model receipt is not fail-closed")
        if receipt.receipt_digest != artifact_digest(receipt._body()):
            raise OfferIdentityPolicyError("model receipt digest mismatch")
        return receipt


def _invalid_receipt(
    role: OfferIdentityModelRole,
    subject: OfferIdentitySubject,
    code: str,
    contract: OfferIdentityContract,
) -> _ModelReceipt:
    partial = _ModelReceipt(
        role=role,
        subject_id=subject.subject_id,
        valid=False,
        decision=OfferIdentityDecision.AMBIGUOUS,
        reason_code=code,
        rationale="",
        input_digest=contract.input_digest,
        policy_digest=contract.policy_digest,
        catalog_digest=contract.catalog_digest,
        subject_digest=subject.subject_digest,
        evidence_refs_digest=subject.evidence_refs_digest,
        reviewed_evidence_ref_digests=(),
        receipt_digest="0" * 64,
    )
    return _ModelReceipt(
        role=partial.role,
        subject_id=partial.subject_id,
        valid=partial.valid,
        decision=partial.decision,
        reason_code=partial.reason_code,
        rationale=partial.rationale,
        input_digest=partial.input_digest,
        policy_digest=partial.policy_digest,
        catalog_digest=partial.catalog_digest,
        subject_digest=partial.subject_digest,
        evidence_refs_digest=partial.evidence_refs_digest,
        reviewed_evidence_ref_digests=partial.reviewed_evidence_ref_digests,
        receipt_digest=artifact_digest(partial._body()),
    )


def _validate_model_decision(
    value: Mapping[str, Any],
    *,
    role: OfferIdentityModelRole,
    subject: OfferIdentitySubject,
    contract: OfferIdentityContract,
) -> _ModelReceipt:
    _require_exact_keys(value, _MODEL_DECISION_KEYS, field="model decision")
    bindings = {
        "version": OFFER_IDENTITY_MODEL_DECISION_VERSION,
        "role": role.value,
        "subject_id": subject.subject_id,
        "input_digest": contract.input_digest,
        "policy_digest": contract.policy_digest,
        "catalog_digest": contract.catalog_digest,
        "subject_digest": subject.subject_digest,
        "evidence_refs_digest": subject.evidence_refs_digest,
    }
    for field, expected in bindings.items():
        if value.get(field) != expected:
            raise OfferIdentityPolicyError(f"model decision {field} binding mismatch")
    reviewed = value.get("reviewed_evidence_ref_digests")
    if not isinstance(reviewed, list) or reviewed != list(subject.evidence_ref_digests):
        raise OfferIdentityPolicyError("model decision did not review exact evidence refs")
    try:
        decision = OfferIdentityDecision(str(value.get("decision") or ""))
        reason = OfferIdentityReasonCode(str(value.get("reason_code") or ""))
    except ValueError as exc:
        raise OfferIdentityPolicyError("model decision enum is invalid") from exc
    if reason not in _REASONS_BY_DECISION[decision]:
        raise OfferIdentityPolicyError("model decision reason contradicts decision")
    rationale = value.get("rationale")
    if not isinstance(rationale, str) or not 1 <= len(rationale) <= 600:
        raise OfferIdentityPolicyError("model decision rationale is invalid")
    partial = _ModelReceipt(
        role=role,
        subject_id=subject.subject_id,
        valid=True,
        decision=decision,
        reason_code=reason.value,
        rationale=rationale,
        input_digest=contract.input_digest,
        policy_digest=contract.policy_digest,
        catalog_digest=contract.catalog_digest,
        subject_digest=subject.subject_digest,
        evidence_refs_digest=subject.evidence_refs_digest,
        reviewed_evidence_ref_digests=subject.evidence_ref_digests,
        receipt_digest="0" * 64,
    )
    return _ModelReceipt(
        role=partial.role,
        subject_id=partial.subject_id,
        valid=partial.valid,
        decision=partial.decision,
        reason_code=partial.reason_code,
        rationale=partial.rationale,
        input_digest=partial.input_digest,
        policy_digest=partial.policy_digest,
        catalog_digest=partial.catalog_digest,
        subject_digest=partial.subject_digest,
        evidence_refs_digest=partial.evidence_refs_digest,
        reviewed_evidence_ref_digests=partial.reviewed_evidence_ref_digests,
        receipt_digest=artifact_digest(partial._body()),
    )


def _index_role_results(
    value: Mapping[str, Any] | BaseException | None,
    *,
    role: OfferIdentityModelRole,
    contract: OfferIdentityContract,
) -> tuple[dict[str, _ModelReceipt], list[str]]:
    subjects = {item.subject_id: item for item in contract.subjects}
    def invalid_role(code: str) -> tuple[dict[str, _ModelReceipt], list[str]]:
        return (
            {
                subject_id: _invalid_receipt(role, subject, code, contract)
                for subject_id, subject in subjects.items()
            },
            [f"{role.value}:{code}"],
        )

    if value is None:
        return invalid_role("role_missing")
    if isinstance(value, BaseException):
        return invalid_role("role_error")
    if not isinstance(value, Mapping):
        return invalid_role("role_malformed")
    try:
        _require_exact_keys(value, _MODEL_BATCH_KEYS, field="model batch")
        bindings = {
            "version": OFFER_IDENTITY_MODEL_BATCH_VERSION,
            "role": role.value,
            "input_digest": contract.input_digest,
            "policy_digest": contract.policy_digest,
            "catalog_digest": contract.catalog_digest,
            "request_contract_digest": _role_request_contract(contract, role)[
                "request_contract_digest"
            ],
            "subject_count": len(contract.subjects),
        }
        for field, expected in bindings.items():
            if value.get(field) != expected:
                raise OfferIdentityPolicyError(
                    f"model batch {field} binding mismatch"
                )
        decisions = value.get("decisions")
        if (
            isinstance(decisions, (str, bytes, Mapping))
            or not isinstance(decisions, Sequence)
            or len(decisions) != len(contract.subjects)
        ):
            raise OfferIdentityPolicyError("model batch decision count mismatch")

        rows_by_subject: dict[str, list[Mapping[str, Any]]] = {}
        for row in decisions:
            if not isinstance(row, Mapping):
                raise OfferIdentityPolicyError("model batch contains malformed row")
            subject_id = row.get("subject_id")
            if not isinstance(subject_id, str) or subject_id not in subjects:
                raise OfferIdentityPolicyError(
                    "model batch contains extra or unbound subject"
                )
            rows_by_subject.setdefault(subject_id, []).append(row)
        if set(rows_by_subject) != set(subjects):
            raise OfferIdentityPolicyError("model batch subject coverage mismatch")
        if any(len(rows) != 1 for rows in rows_by_subject.values()):
            raise OfferIdentityPolicyError("model batch contains duplicate subject")

        receipts: dict[str, _ModelReceipt] = {}
        for subject_id, subject in subjects.items():
            receipts[subject_id] = _validate_model_decision(
                rows_by_subject[subject_id][0],
                role=role,
                subject=subject,
                contract=contract,
            )
        return receipts, []
    except (OfferIdentityPolicyError, TypeError, ValueError):
        # An aggregate role response is one atomic receipt. Partial acceptance
        # would turn omission or duplication into an implicit third policy.
        return invalid_role("role_invalid")


@dataclass(frozen=True, slots=True)
class OfferIdentityResolvedDecision:
    subject_id: str
    offer_id: str
    name: str
    name_role: OfferIdentityNameRole
    decision: OfferIdentityDecision
    standalone: bool
    resolution_code: str
    primary: _ModelReceipt
    critic: _ModelReceipt
    decision_digest: str

    def _body(self) -> dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "offer_id": self.offer_id,
            "name": self.name,
            "name_role": self.name_role.value,
            "decision": self.decision.value,
            "standalone": self.standalone,
            "resolution_code": self.resolution_code,
            "primary": self.primary.as_dict(),
            "critic": self.critic.as_dict(),
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self._body(), "decision_digest": self.decision_digest}

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        subject: OfferIdentitySubject,
        contract: OfferIdentityContract,
    ) -> OfferIdentityResolvedDecision:
        _require_exact_keys(
            value, _RESOLVED_DECISION_KEYS, field="resolved identity decision"
        )
        primary = _ModelReceipt.from_mapping(
            value.get("primary"),
            role=OfferIdentityModelRole.PRIMARY,
            subject=subject,
            contract=contract,
        )
        critic = _ModelReceipt.from_mapping(
            value.get("critic"),
            role=OfferIdentityModelRole.CRITIC,
            subject=subject,
            contract=contract,
        )
        expected_decision, expected_code, expected_standalone = (
            _resolve_subject_outcome(subject, primary, critic)
        )
        standalone = value.get("standalone")
        if not isinstance(standalone, bool):
            raise OfferIdentityPolicyError(
                "resolved identity standalone flag is invalid"
            )
        bindings = {
            "subject_id": subject.subject_id,
            "offer_id": subject.offer_id,
            "name": subject.name,
            "name_role": subject.name_role.value,
            "decision": expected_decision.value,
            "standalone": expected_standalone,
            "resolution_code": expected_code,
        }
        for field, expected in bindings.items():
            if value.get(field) != expected:
                raise OfferIdentityPolicyError(
                    f"resolved identity {field} binding mismatch"
                )
        decision = cls(
            subject_id=subject.subject_id,
            offer_id=subject.offer_id,
            name=subject.name,
            name_role=subject.name_role,
            decision=expected_decision,
            standalone=expected_standalone,
            resolution_code=expected_code,
            primary=primary,
            critic=critic,
            decision_digest=_require_sha256(
                value.get("decision_digest"), field="decision_digest"
            ),
        )
        if decision.decision_digest != artifact_digest(decision._body()):
            raise OfferIdentityPolicyError("resolved identity decision digest mismatch")
        return decision


def _resolve_subject_outcome(
    subject: OfferIdentitySubject,
    primary: _ModelReceipt,
    critic: _ModelReceipt,
) -> tuple[OfferIdentityDecision, str, bool]:
    if is_generic_offer_name(subject.name):
        decision = OfferIdentityDecision.GENERIC_CATEGORY
        code = "static_generic_override"
    elif not primary.valid or not critic.valid:
        decision = OfferIdentityDecision.AMBIGUOUS
        code = "invalid_or_missing_model_result"
    elif primary.decision is not critic.decision:
        decision = OfferIdentityDecision.AMBIGUOUS
        code = "independent_models_disagree"
    else:
        decision = primary.decision
        code = "independent_models_agree"
    standalone = bool(
        decision is OfferIdentityDecision.NAMED_OFFERING
        and primary.valid
        and critic.valid
        and primary.decision is OfferIdentityDecision.NAMED_OFFERING
        and critic.decision is OfferIdentityDecision.NAMED_OFFERING
    )
    return decision, code, standalone


@dataclass(frozen=True, slots=True)
class OfferIdentityPolicyResult:
    input_digest: str
    policy_digest: str
    catalog_digest: str
    decisions: tuple[OfferIdentityResolvedDecision, ...]
    diagnostics: tuple[str, ...]
    output_digest: str
    version: str = OFFER_IDENTITY_RESULT_VERSION

    def _body(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "input_digest": self.input_digest,
            "policy_digest": self.policy_digest,
            "catalog_digest": self.catalog_digest,
            "decision_count": len(self.decisions),
            "standalone_count": sum(item.standalone for item in self.decisions),
            "ambiguous_count": sum(
                item.decision is OfferIdentityDecision.AMBIGUOUS
                for item in self.decisions
            ),
            "decisions": [item.as_dict() for item in self.decisions],
            "diagnostics": list(self.diagnostics),
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self._body(), "output_digest": self.output_digest}

    def standalone_names_by_offer(self) -> dict[str, tuple[str, ...]]:
        grouped: dict[str, list[str]] = {}
        for item in self.decisions:
            if item.standalone:
                grouped.setdefault(item.offer_id, []).append(item.name)
        return {key: tuple(value) for key, value in grouped.items()}

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        catalog: OfferCatalog,
    ) -> OfferIdentityPolicyResult:
        """Parse and revalidate a persisted result against the current catalog."""

        if not isinstance(value, Mapping):
            raise OfferIdentityPolicyError("persisted identity result is malformed")
        _require_exact_keys(value, _RESULT_KEYS, field="identity result")
        contract = build_offer_identity_contract(catalog)
        bindings = {
            "version": OFFER_IDENTITY_RESULT_VERSION,
            "input_digest": contract.input_digest,
            "policy_digest": contract.policy_digest,
            "catalog_digest": contract.catalog_digest,
        }
        for field, expected in bindings.items():
            if value.get(field) != expected:
                raise OfferIdentityPolicyError(
                    f"identity result {field} binding mismatch"
                )
        raw_decisions = value.get("decisions")
        if (
            isinstance(raw_decisions, (str, bytes, Mapping))
            or not isinstance(raw_decisions, Sequence)
            or len(raw_decisions) != len(contract.subjects)
        ):
            raise OfferIdentityPolicyError("identity result subject coverage mismatch")
        decisions: list[OfferIdentityResolvedDecision] = []
        for raw, subject in zip(raw_decisions, contract.subjects, strict=True):
            if not isinstance(raw, Mapping):
                raise OfferIdentityPolicyError(
                    "identity result contains malformed decision"
                )
            decisions.append(
                OfferIdentityResolvedDecision.from_mapping(
                    raw, subject=subject, contract=contract
                )
            )
        raw_diagnostics = value.get("diagnostics")
        if (
            isinstance(raw_diagnostics, (str, bytes, Mapping))
            or not isinstance(raw_diagnostics, Sequence)
            or any(not isinstance(item, str) for item in raw_diagnostics)
        ):
            raise OfferIdentityPolicyError("identity result diagnostics are invalid")
        diagnostics = tuple(raw_diagnostics)
        if diagnostics != tuple(sorted(set(diagnostics))):
            raise OfferIdentityPolicyError(
                "identity result diagnostics are not canonical"
            )
        counters = {
            "decision_count": len(decisions),
            "standalone_count": sum(item.standalone for item in decisions),
            "ambiguous_count": sum(
                item.decision is OfferIdentityDecision.AMBIGUOUS
                for item in decisions
            ),
        }
        for field, expected in counters.items():
            observed = value.get(field)
            if isinstance(observed, bool) or observed != expected:
                raise OfferIdentityPolicyError(
                    f"identity result {field} mismatch"
                )
        result = cls(
            version=OFFER_IDENTITY_RESULT_VERSION,
            input_digest=contract.input_digest,
            policy_digest=contract.policy_digest,
            catalog_digest=contract.catalog_digest,
            decisions=tuple(decisions),
            diagnostics=diagnostics,
            output_digest=_require_sha256(
                value.get("output_digest"), field="output_digest"
            ),
        )
        if result.output_digest != artifact_digest(result._body()):
            raise OfferIdentityPolicyError("identity result output_digest mismatch")
        return result

    def validate_against(self, catalog: OfferCatalog) -> None:
        restored = self.from_mapping(self.as_dict(), catalog=catalog)
        if restored != self:
            raise OfferIdentityPolicyError("identity result is not canonical")


def resolve_offer_identity_policy(
    catalog: OfferCatalog,
    *,
    contract: OfferIdentityContract,
    primary_results: Mapping[str, Any] | BaseException | None,
    critic_results: Mapping[str, Any] | BaseException | None,
) -> OfferIdentityPolicyResult:
    """Reconcile independent results without turning model errors into failure."""

    diagnostics: list[str] = []
    try:
        contract.validate_against(catalog)
    except (OfferIdentityPolicyError, OfferCatalogError, TypeError, ValueError):
        # Rebuild only from the already validated code-owned catalog. Responses
        # bound to the stale/corrupt contract are deliberately discarded.
        contract = build_offer_identity_contract(catalog)
        primary_results = None
        critic_results = None
        diagnostics.append("contract_rebuilt_fail_closed")

    primary, primary_diagnostics = _index_role_results(
        primary_results,
        role=OfferIdentityModelRole.PRIMARY,
        contract=contract,
    )
    critic, critic_diagnostics = _index_role_results(
        critic_results,
        role=OfferIdentityModelRole.CRITIC,
        contract=contract,
    )
    diagnostics.extend(primary_diagnostics)
    diagnostics.extend(critic_diagnostics)

    decisions: list[OfferIdentityResolvedDecision] = []
    for subject in contract.subjects:
        primary_receipt = primary[subject.subject_id]
        critic_receipt = critic[subject.subject_id]
        final, resolution_code, standalone = _resolve_subject_outcome(
            subject, primary_receipt, critic_receipt
        )
        body = {
            "subject_id": subject.subject_id,
            "offer_id": subject.offer_id,
            "name": subject.name,
            "name_role": subject.name_role.value,
            "decision": final.value,
            "standalone": standalone,
            "resolution_code": resolution_code,
            "primary": primary_receipt.as_dict(),
            "critic": critic_receipt.as_dict(),
        }
        decisions.append(
            OfferIdentityResolvedDecision(
                subject_id=subject.subject_id,
                offer_id=subject.offer_id,
                name=subject.name,
                name_role=subject.name_role,
                decision=final,
                standalone=standalone,
                resolution_code=resolution_code,
                primary=primary_receipt,
                critic=critic_receipt,
                decision_digest=artifact_digest(body),
            )
        )
    partial = OfferIdentityPolicyResult(
        input_digest=contract.input_digest,
        policy_digest=contract.policy_digest,
        catalog_digest=contract.catalog_digest,
        decisions=tuple(decisions),
        diagnostics=tuple(sorted(set(diagnostics))),
        output_digest="0" * 64,
    )
    return OfferIdentityPolicyResult(
        input_digest=partial.input_digest,
        policy_digest=partial.policy_digest,
        catalog_digest=partial.catalog_digest,
        decisions=partial.decisions,
        diagnostics=partial.diagnostics,
        output_digest=artifact_digest(partial._body()),
    )


__all__ = [
    "OFFER_IDENTITY_CONTRACT_VERSION",
    "OFFER_IDENTITY_CRITIC_PROMPT_VERSION",
    "OFFER_IDENTITY_CRITIC_SYSTEM_PROMPT",
    "OFFER_IDENTITY_MODEL_BATCH_VERSION",
    "OFFER_IDENTITY_MODEL_DECISION_VERSION",
    "OFFER_IDENTITY_POLICY",
    "OFFER_IDENTITY_POLICY_DIGEST",
    "OFFER_IDENTITY_POLICY_VERSION",
    "OFFER_IDENTITY_PRIMARY_PROMPT_VERSION",
    "OFFER_IDENTITY_PRIMARY_SYSTEM_PROMPT",
    "OFFER_IDENTITY_REQUEST_CONTRACT_VERSION",
    "OFFER_IDENTITY_RESULT_VERSION",
    "OfferIdentityContract",
    "OfferIdentityDecision",
    "OfferIdentityModelRole",
    "OfferIdentityNameRole",
    "OfferIdentityPolicyError",
    "OfferIdentityPolicyResult",
    "OfferIdentityReasonCode",
    "OfferIdentityResolvedDecision",
    "OfferIdentitySubject",
    "build_offer_identity_contract",
    "build_offer_identity_model_request",
    "offer_identity_decision_schema",
    "resolve_offer_identity_policy",
]
