from __future__ import annotations

"""Deterministic fact bindings for model-authored evidence summaries.

The module deliberately does not try to be a general-purpose NLP system.  It
extracts only facts whose entity and metric can be bound to caller-owned alias
registries (or to an explicit default).  Anything ambiguous is left
unextracted, so a later completeness check fails closed instead of guessing.

The intended pipeline is:

1. reconstruct the exact text of a logical source claim;
2. call :func:`extract_fact_bindings` for every evidence-tree child;
3. give the resulting immutable bindings to the model as identifiers;
4. require the model to return the identifiers used by each statement;
5. call :func:`validate_statement_bindings` before accepting the statement.

No source-size, binding-count, or statement-length limit exists here.  Alias
registries are data, not hard-coded client rules, which keeps the contract
usable for arbitrary AI Visibility / GEO / AEO studies in Russian or English.
"""

from collections import defaultdict, deque
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import re
from typing import Any, Iterable, Mapping, Sequence


FACT_BINDING_VERSION = "aiv-atomic-fact-binding-v1"


DEFAULT_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "answer_rate": (
        "answer rate",
        "response rate",
        "доля ответов",
        "частота ответов",
    ),
    "availability": (
        "availability",
        "accessibility",
        "доступность",
        "доступ",
    ),
    "brand_knowledge_rate": (
        "brand knowledge",
        "brand knowledge rate",
        "знание бренда",
        "узнаваемость бренда",
    ),
    "citation_rate": (
        "citation rate",
        "source rate",
        "доля цитирования",
        "доля ссылок",
        "цитируемость",
    ),
    "mention_rate": (
        "mention rate",
        "brand mention rate",
        "share of mentions",
        "доля упоминаний",
        "частота упоминаний",
        "упоминаемость",
    ),
    "product_visibility_rate": (
        "product visibility",
        "product visibility rate",
        "видимость продуктов",
        "продуктовая видимость",
    ),
    "recommendation_rate": (
        "recommendation rate",
        "share of recommendations",
        "доля рекомендаций",
        "частота рекомендаций",
        "рекомендуемость",
    ),
    "rank": (
        "rank",
        "ranking position",
        "position",
        "место",
        "позиция",
    ),
    "sample_size": (
        "sample size",
        "number of answers",
        "число ответов",
        "количество ответов",
        "размер выборки",
    ),
    "score": (
        "score",
        "rating",
        "оценка",
        "балл",
        "рейтинг",
    ),
    "top3_rate": (
        "top-3 rate",
        "top 3 rate",
        "top-3 share",
        "доля топ-3",
        "частота попадания в топ-3",
    ),
}


_STATE_ALIASES: dict[str, tuple[str, ...]] = {
    "available": (
        "available",
        "accessible",
        "доступен",
        "доступна",
        "доступно",
        "доступны",
    ),
    "excluded": (
        "excluded",
        "not eligible",
        "исключен",
        "исключена",
        "исключено",
        "не учитывается",
    ),
    "limited": (
        "limited",
        "partially available",
        "ограничен",
        "ограничена",
        "ограничено",
        "частично доступен",
        "частично доступна",
        "частично доступно",
    ),
    "measured": (
        "measured",
        "observed",
        "измерен",
        "измерена",
        "измерено",
        "наблюдается",
    ),
    "not_found": (
        "not found",
        "absent",
        "не найден",
        "не найдена",
        "не найдено",
        "отсутствует",
    ),
    "unavailable": (
        "unavailable",
        "not measured",
        "no data",
        "n/a",
        "недоступен",
        "недоступна",
        "недоступно",
        "не измерен",
        "не измерена",
        "не измерено",
        "нет данных",
    ),
    "unknown": (
        "unknown",
        "not determined",
        "неизвестно",
        "не определен",
        "не определена",
        "не определено",
    ),
}


_NUMBER = r"[+\-\u2212]?\d+(?:[.,]\d+)?"
_PERCENTAGE_POINT_RE = re.compile(
    rf"(?P<number>{_NUMBER})\s*(?P<unit>п\.?\s*п\.?|p\.?\s*p\.?|percentage\s+points?)",
    re.IGNORECASE,
)
_PERCENT_RE = re.compile(
    rf"(?P<number>{_NUMBER})\s*(?P<unit>%|percent(?:age)?|процент(?:а|ов)?)",
    re.IGNORECASE,
)
_RATIO_RE = re.compile(
    rf"(?P<numerator>{_NUMBER})\s*(?P<separator>/|из|of|out\s+of)\s*"
    rf"(?P<denominator>{_NUMBER})(?:\s*(?P<unit>answers?|responses?|mentions?|"
    r"ответ(?:а|ов)?|упоминани(?:е|я|й)))?",
    re.IGNORECASE,
)
_COUNT_RE = re.compile(
    rf"(?P<number>{_NUMBER})\s*(?P<unit>answers?|responses?|mentions?|citations?|"
    r"ответ(?:а|ов)?|упоминани(?:е|я|й)|ссыл(?:ка|ки|ок))",
    re.IGNORECASE,
)
_ASSIGNED_NUMBER_RE = re.compile(
    rf"(?:=|:)\s*(?P<number>{_NUMBER})(?!\s*(?:%|/|из\b|of\b|out\s+of\b))",
    re.IGNORECASE,
)
_RANK_AFTER_RE = re.compile(
    r"(?P<label>rank|ranking\s+position|position|место|позици[яи])\s*"
    rf"(?:=|:|№|#)?\s*(?P<number>{_NUMBER})",
    re.IGNORECASE,
)
_RANK_BEFORE_RE = re.compile(
    rf"(?P<number>{_NUMBER})\s*(?:-?(?:е|й|я)|st|nd|rd|th)?\s*"
    r"(?P<label>place|position|место|позици[яи])",
    re.IGNORECASE,
)
_TOP_RE = re.compile(
    rf"(?P<label>top|топ)\s*[- ]?\s*(?P<number>{_NUMBER})",
    re.IGNORECASE,
)
_DIRECTION_POSITIVE_RE = re.compile(
    r"(?:\b(?:increase|increased|growth|grew|rise|rose|rising|gain|gained|higher|up)\b|"
    r"\b(?:рост|вырос\w*|увеличил\w*|повысил\w*|выше)\b)",
    re.IGNORECASE,
)
_DIRECTION_NEGATIVE_RE = re.compile(
    r"(?:\b(?:decrease|decreased|decline|declined|fell|drop|dropped|lower|down)\b|"
    r"\b(?:снижен\w*|снизил\w*|падени\w*|упал\w*|ниже)\b)",
    re.IGNORECASE,
)


class FactBindingError(ValueError):
    """Raised when a fact-binding contract cannot be proven."""


@dataclass(frozen=True, slots=True)
class AtomicFactBinding:
    """One immutable, content-addressed semantic relation from exact text."""

    version: str
    binding_id: str
    child_id: str
    claim_id: str
    source_sha256: str
    source_excerpt_sha256: str
    source_char_start: int
    source_char_end: int
    source_utf8_start: int
    source_utf8_end: int
    fact_char_start: int
    fact_char_end: int
    source_order: int
    source_excerpt: str
    fact_lexeme: str
    entity: str
    entity_lexeme: str
    metric: str
    metric_lexeme: str
    kind: str
    state: str | None
    value: str | None
    unit: str | None
    numerator: str | None
    denominator: str | None
    sign: str
    direction: str
    order_relation: str | None
    order_value: str | None

    def semantic_signature(self) -> tuple[str | None, ...]:
        """Return the relation that a model-authored statement must preserve."""

        return (
            self.entity,
            self.metric,
            self.kind,
            self.state,
            self.value,
            self.unit,
            self.numerator,
            self.denominator,
            self.sign,
            self.direction,
            self.order_relation,
            self.order_value,
        )

    def payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "child_id": self.child_id,
            "claim_id": self.claim_id,
            "source_sha256": self.source_sha256,
            "source_excerpt_sha256": self.source_excerpt_sha256,
            "source_char_start": self.source_char_start,
            "source_char_end": self.source_char_end,
            "source_utf8_start": self.source_utf8_start,
            "source_utf8_end": self.source_utf8_end,
            "fact_char_start": self.fact_char_start,
            "fact_char_end": self.fact_char_end,
            "source_order": self.source_order,
            "source_excerpt": self.source_excerpt,
            "fact_lexeme": self.fact_lexeme,
            "entity": self.entity,
            "entity_lexeme": self.entity_lexeme,
            "metric": self.metric,
            "metric_lexeme": self.metric_lexeme,
            "kind": self.kind,
            "state": self.state,
            "value": self.value,
            "unit": self.unit,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "sign": self.sign,
            "direction": self.direction,
            "order_relation": self.order_relation,
            "order_value": self.order_value,
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self.payload(), "binding_id": self.binding_id}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AtomicFactBinding":
        try:
            binding = cls(
                version=str(value["version"]),
                binding_id=str(value["binding_id"]),
                child_id=str(value["child_id"]),
                claim_id=str(value["claim_id"]),
                source_sha256=str(value["source_sha256"]),
                source_excerpt_sha256=str(value["source_excerpt_sha256"]),
                source_char_start=int(value["source_char_start"]),
                source_char_end=int(value["source_char_end"]),
                source_utf8_start=int(value["source_utf8_start"]),
                source_utf8_end=int(value["source_utf8_end"]),
                fact_char_start=int(value["fact_char_start"]),
                fact_char_end=int(value["fact_char_end"]),
                source_order=int(value["source_order"]),
                source_excerpt=str(value["source_excerpt"]),
                fact_lexeme=str(value["fact_lexeme"]),
                entity=str(value["entity"]),
                entity_lexeme=str(value["entity_lexeme"]),
                metric=str(value["metric"]),
                metric_lexeme=str(value["metric_lexeme"]),
                kind=str(value["kind"]),
                state=_optional_string(value.get("state")),
                value=_optional_string(value.get("value")),
                unit=_optional_string(value.get("unit")),
                numerator=_optional_string(value.get("numerator")),
                denominator=_optional_string(value.get("denominator")),
                sign=str(value["sign"]),
                direction=str(value["direction"]),
                order_relation=_optional_string(value.get("order_relation")),
                order_value=_optional_string(value.get("order_value")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise FactBindingError(f"Malformed atomic fact binding: {exc}") from exc
        validate_binding_integrity(binding)
        return binding


@dataclass(frozen=True, slots=True)
class BindingMatch:
    source_binding_id: str
    statement_binding_id: str
    statement_char_start: int
    statement_char_end: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_binding_id": self.source_binding_id,
            "statement_binding_id": self.statement_binding_id,
            "statement_char_start": self.statement_char_start,
            "statement_char_end": self.statement_char_end,
        }


@dataclass(frozen=True, slots=True)
class BindingValidationReport:
    valid: bool
    required_binding_count: int
    statement_binding_count: int
    matches: tuple[BindingMatch, ...]
    missing_binding_ids: tuple[str, ...]
    unexpected_statement_binding_ids: tuple[str, ...]
    order_violations: tuple[str, ...]
    errors: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "required_binding_count": self.required_binding_count,
            "statement_binding_count": self.statement_binding_count,
            "matches": [match.as_dict() for match in self.matches],
            "missing_binding_ids": list(self.missing_binding_ids),
            "unexpected_statement_binding_ids": list(
                self.unexpected_statement_binding_ids
            ),
            "order_violations": list(self.order_violations),
            "errors": list(self.errors),
        }


@dataclass(frozen=True, slots=True)
class _AliasMatch:
    canonical: str
    lexeme: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class _FactCandidate:
    segment_start: int
    segment_end: int
    fact_start: int
    fact_end: int
    entity: str
    entity_lexeme: str
    metric: str
    metric_lexeme: str
    kind: str
    state: str | None = None
    value: str | None = None
    unit: str | None = None
    numerator: str | None = None
    denominator: str | None = None
    sign: str = "not_applicable"
    direction: str = "neutral"
    order_relation: str | None = None
    order_value: str | None = None


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("optional fact field must be a string or null")
    return value


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _binding_id(payload: Mapping[str, Any]) -> str:
    digest = _sha256_text(_canonical_json(payload))
    return f"afb_{digest}"


def _normalize_decimal(value: str) -> str:
    normalized = value.strip().replace("\u2212", "-").replace(",", ".")
    try:
        decimal = Decimal(normalized)
    except InvalidOperation as exc:
        raise FactBindingError(f"Invalid numeric fact value: {value!r}") from exc
    if not decimal.is_finite():
        raise FactBindingError(f"Non-finite numeric fact value: {value!r}")
    result = format(decimal, "f")
    if "." in result:
        result = result.rstrip("0").rstrip(".")
    if result in {"-0", "+0", ""}:
        return "0"
    return result.lstrip("+")


def _numeric_direction(
    raw_value: str,
    context: str,
    *,
    fact_start: int,
    fact_end: int,
) -> str:
    stripped = raw_value.strip().replace("\u2212", "-")
    if stripped.startswith("+"):
        return "positive"
    if stripped.startswith("-"):
        return "negative"

    candidates: list[tuple[tuple[int, int], str]] = []
    for direction, pattern in (
        ("positive", _DIRECTION_POSITIVE_RE),
        ("negative", _DIRECTION_NEGATIVE_RE),
    ):
        for match in pattern.finditer(context):
            if match.end() <= fact_start:
                distance = (0, fact_start - match.end())
            elif match.start() >= fact_end:
                distance = (1, match.start() - fact_end)
            else:
                distance = (0, 0)
            candidates.append((distance, direction))
    if not candidates:
        return "neutral"
    candidates.sort(key=lambda item: item[0])
    best_distance = candidates[0][0]
    best_directions = {
        direction for distance, direction in candidates if distance == best_distance
    }
    if len(best_directions) != 1:
        return "conflict"
    return next(iter(best_directions))


def _numeric_sign(raw_value: str) -> str:
    stripped = raw_value.strip().replace("\u2212", "-")
    if stripped.startswith("+"):
        return "positive"
    if stripped.startswith("-"):
        return "negative"
    return "unsigned"


def _merge_aliases(
    defaults: Mapping[str, Sequence[str]],
    additions: Mapping[str, Sequence[str]] | None,
) -> dict[str, tuple[str, ...]]:
    merged: dict[str, list[str]] = {
        canonical: list(aliases) for canonical, aliases in defaults.items()
    }
    for canonical, aliases in (additions or {}).items():
        if not isinstance(canonical, str) or not canonical.strip():
            raise FactBindingError("Alias canonical names must be non-empty strings")
        merged.setdefault(canonical, [])
        merged[canonical].extend(aliases)

    alias_owner: dict[str, str] = {}
    result: dict[str, tuple[str, ...]] = {}
    for canonical, aliases in merged.items():
        candidates = [canonical.replace("_", " "), *aliases]
        unique: list[str] = []
        seen: set[str] = set()
        for alias in candidates:
            if not isinstance(alias, str) or not alias.strip():
                raise FactBindingError("Aliases must be non-empty strings")
            normalized = alias.strip().casefold()
            owner = alias_owner.get(normalized)
            if owner is not None and owner != canonical:
                raise FactBindingError(
                    f"Ambiguous alias {alias!r} belongs to {owner!r} and {canonical!r}"
                )
            alias_owner[normalized] = canonical
            if normalized not in seen:
                unique.append(alias.strip())
                seen.add(normalized)
        result[canonical] = tuple(unique)
    return result


def _find_alias_matches(
    text: str,
    aliases: Mapping[str, Sequence[str]],
) -> tuple[_AliasMatch, ...]:
    candidates: list[_AliasMatch] = []
    for canonical, variants in aliases.items():
        for variant in variants:
            pattern = re.compile(
                rf"(?<![\w]){re.escape(variant)}(?![\w])",
                re.IGNORECASE | re.UNICODE,
            )
            for match in pattern.finditer(text):
                candidates.append(
                    _AliasMatch(
                        canonical=canonical,
                        lexeme=match.group(0),
                        start=match.start(),
                        end=match.end(),
                    )
                )

    # Longest alias wins when aliases overlap; a canonical spelling must not
    # create a second match inside a more specific phrase.
    selected: list[_AliasMatch] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (item.start, -(item.end - item.start), item.canonical),
    ):
        # Candidates are ordered by start and longest-first at equal starts, so
        # only the last selected interval can overlap the current one.
        if selected and candidate.start < selected[-1].end:
            continue
        selected.append(candidate)
    return tuple(sorted(selected, key=lambda item: (item.start, item.end)))


def _segments(text: str) -> tuple[tuple[int, int], ...]:
    """Split prose without treating decimal punctuation as a boundary."""

    ranges: list[tuple[int, int]] = []
    start = 0
    index = 0
    while index < len(text):
        character = text[index]
        boundary = character in "\n;"
        if character in ".!?" and (
            index + 1 == len(text) or text[index + 1].isspace()
        ):
            boundary = True
        if boundary:
            end = index + 1
            left = start
            while left < end and text[left].isspace():
                left += 1
            right = end
            while right > left and (text[right - 1].isspace() or text[right - 1] in ";"):
                right -= 1
            if left < right:
                ranges.append((left, right))
            start = end
        index += 1
    left = start
    while left < len(text) and text[left].isspace():
        left += 1
    right = len(text)
    while right > left and text[right - 1].isspace():
        right -= 1
    if left < right:
        ranges.append((left, right))
    return tuple(ranges)


def _bucket_matches_by_segment(
    segments: Sequence[tuple[int, int]],
    matches: Sequence[_AliasMatch],
) -> tuple[tuple[_AliasMatch, ...], ...]:
    """Assign sorted, non-overlapping matches to sorted segments in O(n)."""

    buckets: list[tuple[_AliasMatch, ...]] = []
    cursor = 0
    for start, end in segments:
        while cursor < len(matches) and matches[cursor].end <= start:
            cursor += 1
        following = cursor
        bucket: list[_AliasMatch] = []
        while following < len(matches) and matches[following].start < end:
            match = matches[following]
            if start <= match.start < match.end <= end:
                bucket.append(match)
            following += 1
        cursor = following
        buckets.append(tuple(bucket))
    return tuple(buckets)


def _nearest_unambiguous(
    matches: Sequence[_AliasMatch],
    fact_start: int,
    fact_end: int,
) -> _AliasMatch | None:
    if not matches:
        return None

    def distance(match: _AliasMatch) -> tuple[int, int]:
        if match.end <= fact_start:
            return (0, fact_start - match.end)
        if match.start >= fact_end:
            return (1, match.start - fact_end)
        return (0, 0)

    ranked = sorted(matches, key=lambda match: (*distance(match), match.start))
    best_distance = distance(ranked[0])
    tied = [match for match in ranked if distance(match) == best_distance]
    canonicals = {match.canonical for match in tied}
    if len(canonicals) != 1:
        return None
    return ranked[0]


def _default_match(canonical: str, fact_start: int) -> _AliasMatch:
    return _AliasMatch(
        canonical=canonical,
        lexeme=canonical.replace("_", " "),
        start=fact_start,
        end=fact_start,
    )


def _candidate_associations(
    *,
    entity_matches: Sequence[_AliasMatch],
    metric_matches: Sequence[_AliasMatch],
    fact_start: int,
    fact_end: int,
    default_entity: str | None,
    default_metric: str | None,
) -> tuple[_AliasMatch, _AliasMatch] | None:
    entity = _nearest_unambiguous(entity_matches, fact_start, fact_end)
    metric = _nearest_unambiguous(metric_matches, fact_start, fact_end)
    if entity is None and default_entity is not None:
        entity = _default_match(default_entity, fact_start)
    if metric is None and default_metric is not None:
        metric = _default_match(default_metric, fact_start)
    if entity is None or metric is None:
        return None
    return entity, metric


def _state_matches(text: str) -> tuple[_AliasMatch, ...]:
    return _find_alias_matches(text, _STATE_ALIASES)


def _byte_offset(text: str, char_offset: int) -> int:
    return len(text[:char_offset].encode("utf-8"))


def _utf8_offsets_for(text: str, char_offsets: Iterable[int]) -> dict[int, int]:
    """Resolve many UTF-8 offsets in one source scan instead of O(facts * text)."""

    wanted = set(char_offsets)
    if any(offset < 0 or offset > len(text) for offset in wanted):
        raise FactBindingError("Requested UTF-8 offset lies outside source text")
    resolved: dict[int, int] = {}
    utf8_offset = 0
    if 0 in wanted:
        resolved[0] = 0
    for char_offset, character in enumerate(text, start=1):
        utf8_offset += len(character.encode("utf-8"))
        if char_offset in wanted:
            resolved[char_offset] = utf8_offset
    if set(resolved) != wanted:
        raise AssertionError("Internal UTF-8 offset accounting mismatch")
    return resolved


def _make_binding(
    source_text: str,
    *,
    child_id: str,
    claim_id: str,
    source_order: int,
    candidate: _FactCandidate,
    source_sha256: str,
    utf8_offsets: Mapping[int, int],
) -> AtomicFactBinding:
    excerpt = source_text[candidate.segment_start : candidate.segment_end]
    fact_lexeme = source_text[candidate.fact_start : candidate.fact_end]
    payload: dict[str, Any] = {
        "version": FACT_BINDING_VERSION,
        "child_id": child_id,
        "claim_id": claim_id,
        "source_sha256": source_sha256,
        "source_excerpt_sha256": _sha256_text(excerpt),
        "source_char_start": candidate.segment_start,
        "source_char_end": candidate.segment_end,
        "source_utf8_start": utf8_offsets[candidate.segment_start],
        "source_utf8_end": utf8_offsets[candidate.segment_end],
        "fact_char_start": candidate.fact_start,
        "fact_char_end": candidate.fact_end,
        "source_order": source_order,
        "source_excerpt": excerpt,
        "fact_lexeme": fact_lexeme,
        "entity": candidate.entity,
        "entity_lexeme": candidate.entity_lexeme,
        "metric": candidate.metric,
        "metric_lexeme": candidate.metric_lexeme,
        "kind": candidate.kind,
        "state": candidate.state,
        "value": candidate.value,
        "unit": candidate.unit,
        "numerator": candidate.numerator,
        "denominator": candidate.denominator,
        "sign": candidate.sign,
        "direction": candidate.direction,
        "order_relation": candidate.order_relation,
        "order_value": candidate.order_value,
    }
    return AtomicFactBinding(binding_id=_binding_id(payload), **payload)


def extract_fact_bindings(
    source_text: str,
    *,
    child_id: str,
    claim_id: str | None = None,
    entity_aliases: Mapping[str, Sequence[str]],
    metric_aliases: Mapping[str, Sequence[str]] | None = None,
    default_entity: str | None = None,
    default_metric: str | None = None,
) -> tuple[AtomicFactBinding, ...]:
    """Extract conservative, immutable facts from exact source text.

    ``entity_aliases`` is required because client/product/model entity scope is
    owned by the study, not by a language-model guess.  ``default_entity`` and
    ``default_metric`` are useful for already typed table cells.  Defaults must
    be canonical names present in their respective registries.
    """

    if not isinstance(source_text, str):
        raise FactBindingError("source_text must be a string")
    if not isinstance(child_id, str) or not child_id:
        raise FactBindingError("child_id must be a non-empty string")
    resolved_claim_id = claim_id if claim_id is not None else child_id
    if not isinstance(resolved_claim_id, str) or not resolved_claim_id:
        raise FactBindingError("claim_id must be a non-empty string")

    entities = _merge_aliases({}, entity_aliases)
    metrics = _merge_aliases(DEFAULT_METRIC_ALIASES, metric_aliases)
    if default_entity is not None and default_entity not in entities:
        raise FactBindingError("default_entity is absent from entity_aliases")
    if default_metric is not None and default_metric not in metrics:
        raise FactBindingError("default_metric is absent from metric_aliases")

    entity_matches = _find_alias_matches(source_text, entities)
    metric_matches = _find_alias_matches(source_text, metrics)
    states = _state_matches(source_text)
    candidates: list[_FactCandidate] = []

    segment_ranges = _segments(source_text)
    entity_buckets = _bucket_matches_by_segment(segment_ranges, entity_matches)
    metric_buckets = _bucket_matches_by_segment(segment_ranges, metric_matches)
    state_buckets = _bucket_matches_by_segment(segment_ranges, states)

    for (
        (segment_start, segment_end),
        segment_entities,
        segment_metrics,
        segment_states,
    ) in zip(
        segment_ranges,
        entity_buckets,
        metric_buckets,
        state_buckets,
        strict=True,
    ):
        segment = source_text[segment_start:segment_end]
        occupied: list[tuple[int, int]] = []

        def add_numeric(
            match: re.Match[str],
            *,
            unit: str,
            kind: str = "metric_value",
            numerator: str | None = None,
            denominator: str | None = None,
            order_relation: str | None = None,
            order_value: str | None = None,
        ) -> None:
            fact_start = segment_start + match.start()
            fact_end = segment_start + match.end()
            if any(fact_start < end and start < fact_end for start, end in occupied):
                return
            association = _candidate_associations(
                entity_matches=segment_entities,
                metric_matches=segment_metrics,
                fact_start=fact_start,
                fact_end=fact_end,
                default_entity=default_entity,
                default_metric=default_metric,
            )
            if association is None:
                return
            entity, metric = association
            raw_number = match.groupdict().get("number")
            normalized_numerator = (
                _normalize_decimal(numerator) if numerator is not None else None
            )
            normalized_denominator = (
                _normalize_decimal(denominator) if denominator is not None else None
            )
            value = (
                _normalize_decimal(raw_number)
                if raw_number is not None
                else normalized_numerator
            )
            candidates.append(
                _FactCandidate(
                    segment_start=segment_start,
                    segment_end=segment_end,
                    fact_start=fact_start,
                    fact_end=fact_end,
                    entity=entity.canonical,
                    entity_lexeme=entity.lexeme,
                    metric=("rank" if order_relation == "rank" else metric.canonical),
                    metric_lexeme=(
                        match.groupdict().get("label") or metric.lexeme
                    ),
                    kind=kind,
                    value=value,
                    unit=unit,
                    numerator=normalized_numerator,
                    denominator=normalized_denominator,
                    sign=_numeric_sign(raw_number or numerator or ""),
                    direction=_numeric_direction(
                        raw_number or numerator or "",
                        segment,
                        fact_start=match.start(),
                        fact_end=match.end(),
                    ),
                    order_relation=order_relation,
                    order_value=(
                        _normalize_decimal(order_value)
                        if order_value is not None
                        else None
                    ),
                )
            )
            occupied.append((fact_start, fact_end))

        for pattern in (_RANK_AFTER_RE, _RANK_BEFORE_RE):
            for match in pattern.finditer(segment):
                number = match.group("number")
                add_numeric(
                    match,
                    unit="rank",
                    kind="order",
                    order_relation="rank",
                    order_value=number,
                )
        for match in _TOP_RE.finditer(segment):
            number = match.group("number")
            add_numeric(
                match,
                unit="rank_limit",
                kind="order",
                order_relation="top_n",
                order_value=number,
            )
        for match in _PERCENTAGE_POINT_RE.finditer(segment):
            add_numeric(match, unit="percentage_point")
        for match in _PERCENT_RE.finditer(segment):
            add_numeric(match, unit="percent")
        for match in _RATIO_RE.finditer(segment):
            add_numeric(
                match,
                unit="ratio",
                numerator=match.group("numerator"),
                denominator=match.group("denominator"),
            )
        for match in _COUNT_RE.finditer(segment):
            add_numeric(match, unit="count")
        for match in _ASSIGNED_NUMBER_RE.finditer(segment):
            add_numeric(match, unit="number")

        for state in segment_states:
            association = _candidate_associations(
                entity_matches=segment_entities,
                metric_matches=segment_metrics,
                fact_start=state.start,
                fact_end=state.end,
                default_entity=default_entity,
                default_metric=default_metric,
            )
            if association is None:
                continue
            entity, metric = association
            candidates.append(
                _FactCandidate(
                    segment_start=segment_start,
                    segment_end=segment_end,
                    fact_start=state.start,
                    fact_end=state.end,
                    entity=entity.canonical,
                    entity_lexeme=entity.lexeme,
                    metric=metric.canonical,
                    metric_lexeme=metric.lexeme,
                    kind="metric_state",
                    state=state.canonical,
                )
            )

    # Stable source order is semantic context.  It is checked independently for
    # each child during statement validation, so facts cannot silently trade
    # places inside a ranked/table-like child.
    candidates.sort(
        key=lambda item: (
            item.fact_start,
            item.fact_end,
            item.kind,
            item.entity,
            item.metric,
        )
    )
    source_sha256 = _sha256_text(source_text)
    utf8_offsets = _utf8_offsets_for(
        source_text,
        (
            offset
            for candidate in candidates
            for offset in (candidate.segment_start, candidate.segment_end)
        ),
    )
    bindings = tuple(
        _make_binding(
            source_text,
            child_id=child_id,
            claim_id=resolved_claim_id,
            source_order=index,
            candidate=candidate,
            source_sha256=source_sha256,
            utf8_offsets=utf8_offsets,
        )
        for index, candidate in enumerate(candidates)
    )
    validate_binding_set_integrity(bindings, source_text=source_text)
    return bindings


def validate_binding_integrity(
    binding: AtomicFactBinding,
    *,
    source_text: str | None = None,
) -> None:
    """Verify content identity and, when supplied, the exact source spans."""

    if binding.version != FACT_BINDING_VERSION:
        raise FactBindingError(f"Unsupported fact-binding version: {binding.version}")
    if _binding_id(binding.payload()) != binding.binding_id:
        raise FactBindingError("Fact binding identity mismatch")
    if _sha256_text(binding.source_excerpt) != binding.source_excerpt_sha256:
        raise FactBindingError("Fact binding excerpt digest mismatch")
    if not (0 <= binding.source_char_start <= binding.source_char_end):
        raise FactBindingError("Invalid source character span")
    if not (
        binding.source_char_start
        <= binding.fact_char_start
        <= binding.fact_char_end
        <= binding.source_char_end
    ):
        raise FactBindingError("Fact lexeme lies outside its source excerpt")
    relative_start = binding.fact_char_start - binding.source_char_start
    relative_end = binding.fact_char_end - binding.source_char_start
    if binding.source_excerpt[relative_start:relative_end] != binding.fact_lexeme:
        raise FactBindingError("Fact lexeme does not match its source span")
    if binding.source_utf8_start > binding.source_utf8_end:
        raise FactBindingError("Invalid source UTF-8 span")
    if binding.source_utf8_end - binding.source_utf8_start != len(
        binding.source_excerpt.encode("utf-8")
    ):
        raise FactBindingError("Fact binding source UTF-8 length mismatch")

    if source_text is None:
        return
    if _sha256_text(source_text) != binding.source_sha256:
        raise FactBindingError("Fact binding source digest mismatch")
    if source_text[binding.source_char_start : binding.source_char_end] != binding.source_excerpt:
        raise FactBindingError("Fact binding source excerpt mismatch")
    if source_text[binding.fact_char_start : binding.fact_char_end] != binding.fact_lexeme:
        raise FactBindingError("Fact binding source lexeme mismatch")
    if _byte_offset(source_text, binding.source_char_start) != binding.source_utf8_start:
        raise FactBindingError("Fact binding source UTF-8 start mismatch")
    if _byte_offset(source_text, binding.source_char_end) != binding.source_utf8_end:
        raise FactBindingError("Fact binding source UTF-8 end mismatch")


def validate_binding_set_integrity(
    bindings: Sequence[AtomicFactBinding],
    *,
    source_text: str,
) -> None:
    """Verify an arbitrary number of bindings against one source in linear time."""

    source_sha256 = _sha256_text(source_text)
    offsets = _utf8_offsets_for(
        source_text,
        (
            offset
            for binding in bindings
            for offset in (binding.source_char_start, binding.source_char_end)
        ),
    )
    seen: set[str] = set()
    for binding in bindings:
        validate_binding_integrity(binding)
        if binding.binding_id in seen:
            raise FactBindingError(f"Duplicate source binding: {binding.binding_id}")
        seen.add(binding.binding_id)
        if binding.source_sha256 != source_sha256:
            raise FactBindingError("Fact binding source digest mismatch")
        if (
            source_text[binding.source_char_start : binding.source_char_end]
            != binding.source_excerpt
        ):
            raise FactBindingError("Fact binding source excerpt mismatch")
        if (
            source_text[binding.fact_char_start : binding.fact_char_end]
            != binding.fact_lexeme
        ):
            raise FactBindingError("Fact binding source lexeme mismatch")
        if offsets[binding.source_char_start] != binding.source_utf8_start:
            raise FactBindingError("Fact binding source UTF-8 start mismatch")
        if offsets[binding.source_char_end] != binding.source_utf8_end:
            raise FactBindingError("Fact binding source UTF-8 end mismatch")


def _binding_aliases(
    bindings: Sequence[AtomicFactBinding],
    *,
    additions: Mapping[str, Sequence[str]] | None,
    attribute: str,
    lexeme_attribute: str,
) -> dict[str, tuple[str, ...]]:
    collected: dict[str, list[str]] = defaultdict(list)
    for binding in bindings:
        collected[getattr(binding, attribute)].append(
            getattr(binding, lexeme_attribute)
        )
    for canonical, aliases in (additions or {}).items():
        collected[canonical].extend(aliases)
    return {canonical: tuple(aliases) for canonical, aliases in collected.items()}


def audit_statement_bindings(
    statement: str,
    *,
    bindings: Sequence[AtomicFactBinding],
    required_binding_ids: Sequence[str],
    entity_aliases: Mapping[str, Sequence[str]] | None = None,
    metric_aliases: Mapping[str, Sequence[str]] | None = None,
    reject_unbound_facts: bool = True,
    enforce_child_order: bool = True,
) -> BindingValidationReport:
    """Audit a model-authored statement against explicit source bindings.

    Matching uses complete semantic signatures and consumes each extracted
    statement fact once.  This is the central anti-cross-binding guarantee: a
    value associated with Gemini cannot satisfy a ChatGPT binding merely because
    both names and both values occur somewhere in the paragraph.
    """

    if not isinstance(statement, str):
        raise FactBindingError("statement must be a string")
    binding_by_id: dict[str, AtomicFactBinding] = {}
    for binding in bindings:
        validate_binding_integrity(binding)
        if binding.binding_id in binding_by_id:
            raise FactBindingError(f"Duplicate source binding: {binding.binding_id}")
        binding_by_id[binding.binding_id] = binding

    required_ids = tuple(required_binding_ids)
    if len(set(required_ids)) != len(required_ids):
        raise FactBindingError("Duplicate required binding identifier")
    unknown = [binding_id for binding_id in required_ids if binding_id not in binding_by_id]
    if unknown:
        raise FactBindingError(f"Unknown required binding identifiers: {unknown}")
    required = tuple(binding_by_id[binding_id] for binding_id in required_ids)
    if not required:
        if statement.strip() and reject_unbound_facts:
            return BindingValidationReport(
                valid=False,
                required_binding_count=0,
                statement_binding_count=0,
                matches=(),
                missing_binding_ids=(),
                unexpected_statement_binding_ids=(),
                order_violations=(),
                errors=("A non-empty statement has no declared fact bindings",),
            )
        return BindingValidationReport(
            valid=True,
            required_binding_count=0,
            statement_binding_count=0,
            matches=(),
            missing_binding_ids=(),
            unexpected_statement_binding_ids=(),
            order_violations=(),
            errors=(),
        )

    # Parse against every known binding, not only the declared subset.  This
    # lets the strict mode catch a model adding a typed fact about another
    # known entity while citing only the convenient binding it wants checked.
    known_bindings = tuple(binding_by_id.values())
    statement_entities = _binding_aliases(
        known_bindings,
        additions=entity_aliases,
        attribute="entity",
        lexeme_attribute="entity_lexeme",
    )
    statement_metrics = _binding_aliases(
        known_bindings,
        additions=metric_aliases,
        attribute="metric",
        lexeme_attribute="metric_lexeme",
    )
    statement_bindings = extract_fact_bindings(
        statement,
        child_id="model_statement",
        claim_id="model_statement",
        entity_aliases=statement_entities,
        metric_aliases=statement_metrics,
    )

    candidates_by_signature: dict[
        tuple[str | None, ...], deque[AtomicFactBinding]
    ] = defaultdict(deque)
    for statement_binding in statement_bindings:
        candidates_by_signature[statement_binding.semantic_signature()].append(
            statement_binding
        )

    matches: list[BindingMatch] = []
    matched_statement_ids: set[str] = set()
    statement_position_by_source_id: dict[str, int] = {}
    missing: list[str] = []
    for source_binding in sorted(
        required,
        key=lambda item: (item.child_id, item.source_order, item.binding_id),
    ):
        queue = candidates_by_signature[source_binding.semantic_signature()]
        while queue and queue[0].binding_id in matched_statement_ids:
            queue.popleft()
        if not queue:
            missing.append(source_binding.binding_id)
            continue
        statement_binding = queue.popleft()
        matched_statement_ids.add(statement_binding.binding_id)
        statement_position_by_source_id[source_binding.binding_id] = (
            statement_binding.fact_char_start
        )
        matches.append(
            BindingMatch(
                source_binding_id=source_binding.binding_id,
                statement_binding_id=statement_binding.binding_id,
                statement_char_start=statement_binding.fact_char_start,
                statement_char_end=statement_binding.fact_char_end,
            )
        )

    unexpected = tuple(
        binding.binding_id
        for binding in statement_bindings
        if binding.binding_id not in matched_statement_ids
    )
    order_violations: list[str] = []
    if enforce_child_order:
        by_child: dict[str, list[AtomicFactBinding]] = defaultdict(list)
        for binding in required:
            if binding.binding_id in statement_position_by_source_id:
                by_child[binding.child_id].append(binding)
        for child_id, child_bindings in by_child.items():
            ordered = sorted(child_bindings, key=lambda item: item.source_order)
            positions = [
                statement_position_by_source_id[binding.binding_id]
                for binding in ordered
            ]
            if positions != sorted(positions):
                order_violations.append(child_id)

    errors: list[str] = []
    if missing:
        errors.append(
            f"The statement does not preserve {len(missing)} required fact binding(s)"
        )
    if reject_unbound_facts and unexpected:
        errors.append(
            f"The statement contains {len(unexpected)} unbound typed fact(s)"
        )
    if order_violations:
        errors.append(
            "The statement changes fact order inside child evidence: "
            + ", ".join(order_violations)
        )
    return BindingValidationReport(
        valid=not errors,
        required_binding_count=len(required),
        statement_binding_count=len(statement_bindings),
        matches=tuple(matches),
        missing_binding_ids=tuple(missing),
        unexpected_statement_binding_ids=(unexpected if reject_unbound_facts else ()),
        order_violations=tuple(order_violations),
        errors=tuple(errors),
    )


def validate_statement_bindings(
    statement: str,
    *,
    bindings: Sequence[AtomicFactBinding],
    required_binding_ids: Sequence[str],
    entity_aliases: Mapping[str, Sequence[str]] | None = None,
    metric_aliases: Mapping[str, Sequence[str]] | None = None,
    reject_unbound_facts: bool = True,
    enforce_child_order: bool = True,
) -> BindingValidationReport:
    """Return a valid report or raise :class:`FactBindingError` fail closed."""

    report = audit_statement_bindings(
        statement,
        bindings=bindings,
        required_binding_ids=required_binding_ids,
        entity_aliases=entity_aliases,
        metric_aliases=metric_aliases,
        reject_unbound_facts=reject_unbound_facts,
        enforce_child_order=enforce_child_order,
    )
    if not report.valid:
        raise FactBindingError("; ".join(report.errors))
    return report


def binding_references(
    bindings: Iterable[AtomicFactBinding],
) -> tuple[dict[str, Any], ...]:
    """Return the minimal immutable contract safe to send to an LLM."""

    references: list[dict[str, Any]] = []
    seen: set[str] = set()
    for binding in bindings:
        validate_binding_integrity(binding)
        if binding.binding_id in seen:
            raise FactBindingError(f"Duplicate source binding: {binding.binding_id}")
        seen.add(binding.binding_id)
        references.append(
            {
                "binding_id": binding.binding_id,
                "child_id": binding.child_id,
                "source_order": binding.source_order,
                "source_excerpt": binding.source_excerpt,
                "entity": binding.entity,
                "metric": binding.metric,
                "kind": binding.kind,
                "state": binding.state,
                "value": binding.value,
                "unit": binding.unit,
                "numerator": binding.numerator,
                "denominator": binding.denominator,
                "sign": binding.sign,
                "direction": binding.direction,
                "order_relation": binding.order_relation,
                "order_value": binding.order_value,
            }
        )
    return tuple(references)


__all__ = [
    "AtomicFactBinding",
    "BindingMatch",
    "BindingValidationReport",
    "DEFAULT_METRIC_ALIASES",
    "FACT_BINDING_VERSION",
    "FactBindingError",
    "audit_statement_bindings",
    "binding_references",
    "extract_fact_bindings",
    "validate_binding_integrity",
    "validate_binding_set_integrity",
    "validate_statement_bindings",
]
