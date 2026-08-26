"""Versioned Russian editorial policy for every reader-facing AIV text.

The production service must not read the author's Yandex Disk.  This module
loads the reviewed, byte-for-byte snapshot checked into ``app/policies`` and
fails closed if that snapshot drifts without a version/checksum update.

The deterministic linter is intentionally conservative.  It reports wording
that the canonical policy forbids unambiguously, but it never rewrites copy or
facts.  Semantic qualities such as a correctly named actor, the carrier of a
percentage, or the strength of a claim still require the editorial model and
its fact-preservation gate.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

LIVE_RUSSIAN_POLICY_VERSION = "live-russian-2026-07-29.1"
LIVE_RUSSIAN_POLICY_SOURCE_DATE = "2026-07-29"
LIVE_RUSSIAN_POLICY_SNAPSHOT = "live_russian_ru.v2026-07-29.md"
LIVE_RUSSIAN_POLICY_SHA256 = (
    "0cd7bbc6cdb006331b3df3c414cc0cdb9bc9860dfa2706b098fe610778392d84"
)

PolicyContext = Literal["report", "interface", "technical", "generic"]
IssueSeverity = Literal["error", "warning"]


@dataclass(frozen=True, slots=True)
class LiveRussianPolicyManifest:
    """Stable identity written to editorial audit artifacts."""

    version: str
    source_date: str
    snapshot: str
    sha256: str
    language: str = "ru"

    def as_dict(self) -> dict[str, str]:
        return {
            "version": self.version,
            "source_date": self.source_date,
            "snapshot": self.snapshot,
            "sha256": self.sha256,
            "language": self.language,
        }


LIVE_RUSSIAN_POLICY_MANIFEST = LiveRussianPolicyManifest(
    version=LIVE_RUSSIAN_POLICY_VERSION,
    source_date=LIVE_RUSSIAN_POLICY_SOURCE_DATE,
    snapshot=f"app/policies/{LIVE_RUSSIAN_POLICY_SNAPSHOT}",
    sha256=LIVE_RUSSIAN_POLICY_SHA256,
)

# Public reports retain the exact policy descriptor that reviewed their copy.
# Keep every still-readable descriptor in this explicit archive when introducing
# a new current policy.  Historical validation never trusts a descriptor merely
# because it is self-sealed in the database: the descriptor must be listed here
# and its checked-in snapshot bytes must still match the archived SHA-256.
_LIVE_RUSSIAN_POLICY_2026_07_29_1 = LiveRussianPolicyManifest(
    version="live-russian-2026-07-29.1",
    source_date="2026-07-29",
    snapshot="app/policies/live_russian_ru.v2026-07-29.md",
    sha256="0cd7bbc6cdb006331b3df3c414cc0cdb9bc9860dfa2706b098fe610778392d84",
)
TRUSTED_LIVE_RUSSIAN_POLICY_MANIFESTS: tuple[LiveRussianPolicyManifest, ...] = tuple(
    dict.fromkeys(
        (
            _LIVE_RUSSIAN_POLICY_2026_07_29_1,
            LIVE_RUSSIAN_POLICY_MANIFEST,
        )
    )
)


def trusted_live_russian_policy_manifest(
    value: Any,
) -> LiveRussianPolicyManifest | None:
    """Resolve one code-owned policy descriptor and verify retained bytes."""

    if not isinstance(value, Mapping):
        return None
    descriptor = dict(value)
    repository_root = Path(__file__).resolve().parents[2]
    policy_root = (repository_root / "app" / "policies").resolve()
    for candidate in TRUSTED_LIVE_RUSSIAN_POLICY_MANIFESTS:
        if descriptor != candidate.as_dict():
            continue
        snapshot_path = (repository_root / candidate.snapshot).resolve()
        if not snapshot_path.is_relative_to(policy_root) or not snapshot_path.is_file():
            return None
        try:
            payload = snapshot_path.read_bytes()
        except OSError:
            return None
        if sha256(payload).hexdigest() != candidate.sha256:
            return None
        return candidate
    return None


@dataclass(frozen=True, slots=True)
class RussianCopyIssue:
    """One deterministic finding; offsets refer to the individual string."""

    code: str
    message: str
    severity: IssueSeverity
    start: int
    end: int
    snippet: str
    path: str = "$"


@dataclass(frozen=True, slots=True)
class RussianCopyLintReport:
    """Bounded output from a full linear scan of registered reader copy."""

    policy_version: str
    policy_sha256: str
    checked_characters: int
    checked_fields: int
    issues: tuple[RussianCopyIssue, ...]
    blocking_issue_count: int = 0
    omitted_issue_count: int = 0
    skipped_paths: tuple[str, ...] = ()

    @property
    def blocking(self) -> bool:
        return self.blocking_issue_count > 0

    @property
    def blocking_count(self) -> int:
        return self.blocking_issue_count

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "policy_sha256": self.policy_sha256,
            "checked_characters": self.checked_characters,
            "checked_fields": self.checked_fields,
            "blocking": self.blocking,
            "blocking_count": self.blocking_count,
            "omitted_issue_count": self.omitted_issue_count,
            "skipped_paths": list(self.skipped_paths),
            "issues": [
                {
                    "code": issue.code,
                    "message": issue.message,
                    "severity": issue.severity,
                    "start": issue.start,
                    "end": issue.end,
                    "snippet": issue.snippet,
                    "path": issue.path,
                }
                for issue in self.issues
            ],
        }


@dataclass(frozen=True, slots=True)
class _PatternRule:
    code: str
    pattern: re.Pattern[str]
    message: str
    severity: IssueSeverity


_BLOCKING_RULES: tuple[_PatternRule, ...] = (
    _PatternRule(
        "long_dash",
        re.compile("—"),
        "Перепишите фразу без длинного тире: для отчётов AIV это явный запрет.",
        "error",
    ),
    _PatternRule(
        "three_dots",
        re.compile(r"\.\.\."),
        "Используйте один знак многоточия или закончите мысль точкой.",
        "error",
    ),
    _PatternRule(
        "straight_quotes",
        re.compile(r'"[^"\n]{1,240}"'),
        "Замените прямые кавычки на «ёлочки», вложенные кавычки на „лапки“.",
        "error",
    ),
    _PatternRule(
        "space_before_punctuation",
        re.compile(r"[ \t]+[,.;:!?]"),
        "Уберите пробел перед знаком препинания.",
        "error",
    ),
    _PatternRule(
        "double_space",
        re.compile(r"(?m)(?<!^) {2,}(?!$)"),
        "Оставьте один пробел.",
        "error",
    ),
)

_MACHINE_PHRASES: tuple[tuple[str, str], ...] = (
    ("modern_world_intro", r"\bв современном мире\b"),
    ("digital_era_intro", r"\bв эпоху цифровизац(?:ии|ии,)\b"),
    ("today_more_than_ever", r"\bсегодня,? как никогда\b"),
    ("summary_thus", r"\bтаким образом\b"),
    ("summary_wrap_up", r"\bподводя итог\b"),
    ("summary_as_we_see", r"\bкак мы видим\b"),
    ("importance_meta", r"\bважно отметить(?:,? что)?\b"),
    ("emphasis_meta", r"\bстоит подчеркнуть(?:,? что)?\b"),
    ("understanding_meta", r"\bследует понимать(?:,? что)?\b"),
    ("lets_examine", r"\bдавайте разбер[её]мся\b"),
    ("dive_into", r"\bпогрузим(?:ся| читателя)\b"),
    ("reveal_secrets", r"\bраскроем секреты\b"),
    ("not_just", r"\bэто не просто\b"),
    ("obvious_three_slices", r"\bтри независимых среза\b"),
    ("obvious_zero_axis", r"\bось начинается с нуля\b"),
    ("obvious_chart_meta", r"\bна (?:этой )?диаграмме (?:видно|показано)\b"),
    ("obvious_legend_instruction", r"\bнажмите (?:на )?легенду,? чтобы\b"),
)

_WARNING_RULES: tuple[_PatternRule, ...] = tuple(
    _PatternRule(
        code,
        re.compile(pattern, re.IGNORECASE),
        "Уберите метаповествование или машинный штамп и начните с факта.",
        "warning",
    )
    for code, pattern in _MACHINE_PHRASES
) + (
    _PatternRule(
        "bureaucratic_is",
        re.compile(r"\bявля(?:ется|ются|лся|лась|лось|лись)\b", re.IGNORECASE),
        "Назовите связь прямо; форма «является» обычно прячет действие.",
        "warning",
    ),
    _PatternRule(
        "bureaucratic_execute",
        re.compile(r"\bосуществля\w*\b", re.IGNORECASE),
        "Назовите исполнителя и действие обычным глаголом.",
        "warning",
    ),
    _PatternRule(
        "bureaucratic_current_time",
        re.compile(r"\b(?:в настоящее время|на сегодняшний день)\b", re.IGNORECASE),
        "Замените канцелярскую связку конкретной датой, периодом или словом «сейчас».",
        "warning",
    ),
)

_ALL_RULES = _BLOCKING_RULES + _WARNING_RULES

DEFAULT_EXCLUDED_READER_SUBTREES = frozenset(
    {
        "raw",
        "raw_data",
        "raw_answer",
        "raw_answers",
        "evidence",
        "citations",
        "sources",
        "provenance",
    }
)
DEFAULT_EXCLUDED_READER_KEYS = frozenset(
    {
        "answer",
        "answer_text",
        "quote",
        "source_url",
        "url",
        "href",
        "sha256",
        "digest",
        "id",
        "run_id",
    }
)

# These rules are deliberately path-aware.  A global ban on fields named
# ``title`` or ``name`` would hide report headings, finding titles and other
# authored copy.  Only literal client/source identity branches are skipped.
_LITERAL_READER_PATH_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"^\$\.published_report\.brand\."
        r"(?:name|products|offers|entity_scope)(?:\.|\[|$)"
    ),
    re.compile(r"^\$\.published_report\.competitors\[\d+\]\.name$"),
    re.compile(
        r"^\$\.published_report\.technical\."
        r"(?:pages|page_details|audited_pages)\[\d+\]\.title$"
    ),
    re.compile(r"^\$\.(?:brand|client|customer)\.name$"),
)


def _is_literal_reader_path(path: str) -> bool:
    return any(pattern.search(path) for pattern in _LITERAL_READER_PATH_PATTERNS)


_CONTEXT_INSTRUCTIONS: dict[PolicyContext, str] = {
    "report": (
        "Жанр: аналитический отчёт. Называй носителя каждой метрики, отделяй "
        "наблюдение от вывода и указывай источник данных."
    ),
    "interface": (
        "Жанр: интерфейс. Пиши коротко; сообщение об ошибке называет причину и "
        "следующий шаг."
    ),
    "technical": (
        "Жанр: техническое объяснение. Субъектами выступают система и читатель; "
        "не упрощай точные термины."
    ),
    "generic": "Жанр не задан. Выбери дозировку правил по назначению текста.",
}


def _snapshot_path() -> Path:
    return (
        Path(__file__).resolve().parents[1] / "policies" / LIVE_RUSSIAN_POLICY_SNAPSHOT
    )


@lru_cache(maxsize=1)
def load_live_russian_policy() -> str:
    """Load the bundled snapshot and reject unversioned policy drift."""

    snapshot_path = _snapshot_path()
    try:
        payload = snapshot_path.read_bytes()
    except OSError as exc:  # pragma: no cover - deployment packaging failure
        raise RuntimeError(
            f"Russian policy snapshot is unavailable: {snapshot_path}"
        ) from exc
    actual_sha256 = sha256(payload).hexdigest()
    if actual_sha256 != LIVE_RUSSIAN_POLICY_SHA256:
        raise RuntimeError(
            "Russian policy snapshot checksum mismatch: "
            f"expected {LIVE_RUSSIAN_POLICY_SHA256}, got {actual_sha256}"
        )
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:  # pragma: no cover - checksum pins UTF-8 bytes
        raise RuntimeError("Russian policy snapshot is not valid UTF-8") from exc


def assert_live_russian_policy_integrity() -> LiveRussianPolicyManifest:
    """Eager deployment/startup check; returns the audit-ready manifest."""

    load_live_russian_policy()
    return LIVE_RUSSIAN_POLICY_MANIFEST


def build_live_russian_policy_prompt(
    *,
    context: PolicyContext = "report",
    preserve_facts: bool = True,
) -> str:
    """Return the canonical rules plus AIV's fact-safety integration contract."""

    if context not in _CONTEXT_INSTRUCTIONS:
        raise ValueError(f"Unsupported Russian policy context: {context!r}")
    fact_rule = (
        "Не меняй числа, имена, URL, роли, причинно-следственные связи, степень "
        "уверенности и силу утверждения. Если живой вариант искажает факт, оставь "
        "точную исходную формулировку."
        if preserve_facts
        else "Сохраняй смысл и силу каждого утверждения."
    )
    integration = "\n".join(
        (
            "# Контракт редактора AIV",
            f"Политика: {LIVE_RUSSIAN_POLICY_VERSION}",
            f"SHA-256: {LIVE_RUSSIAN_POLICY_SHA256}",
            _CONTEXT_INSTRUCTIONS[context],
            fact_rule,
            (
                "Не используй длинное тире в тексте AIV. Если в каноническом "
                "snapshot ниже встречается общее типографическое правило о тире, "
                "явный запрет раздела 0 и это продуктовое правило имеют приоритет."
            ),
            (
                "Не добавляй метакомментарии о диаграмме, оси, количестве срезов "
                "или очевидной структуре блока."
            ),
            "Верни только данные в запрошенной схеме, без отчёта о редактуре.",
        )
    )
    return f"{integration}\n\n# Канонические правила\n\n{load_live_russian_policy()}"


def _snippet(text: str, start: int, end: int, radius: int = 42) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    return text[left:right].replace("\n", " ").strip()


def lint_russian_copy(
    text: str,
    *,
    path: str = "$",
    max_issues: int = 200,
) -> RussianCopyLintReport:
    """Scan one reader-facing string without rewriting or truncating the input."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if max_issues < 1:
        raise ValueError("max_issues must be at least 1")

    found: list[RussianCopyIssue] = []
    seen: set[tuple[str, int, int]] = set()
    for rule in _ALL_RULES:
        for match in rule.pattern.finditer(text):
            identity = (rule.code, match.start(), match.end())
            if identity in seen:
                continue
            seen.add(identity)
            found.append(
                RussianCopyIssue(
                    code=rule.code,
                    message=rule.message,
                    severity=rule.severity,
                    start=match.start(),
                    end=match.end(),
                    snippet=_snippet(text, match.start(), match.end()),
                    path=path,
                )
            )
    found.sort(key=lambda issue: (issue.start, issue.end, issue.code))
    omitted = max(0, len(found) - max_issues)
    return RussianCopyLintReport(
        policy_version=LIVE_RUSSIAN_POLICY_VERSION,
        policy_sha256=LIVE_RUSSIAN_POLICY_SHA256,
        checked_characters=len(text),
        checked_fields=1,
        issues=tuple(found[:max_issues]),
        blocking_issue_count=sum(issue.severity == "error" for issue in found),
        omitted_issue_count=omitted,
    )


def _json_path(parent: str, key: str | int) -> str:
    if isinstance(key, int):
        return f"{parent}[{key}]"
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
        return f"{parent}.{key}"
    escaped = key.replace("\\", "\\\\").replace("'", "\\'")
    return f"{parent}['{escaped}']"


def _iter_reader_copy(
    value: Any,
    *,
    path: str,
    excluded_subtrees: frozenset[str],
    excluded_keys: frozenset[str],
) -> Iterator[tuple[str, str] | tuple[str, None]]:
    if isinstance(value, str):
        yield path, value
        return
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            child_path = _json_path(path, key)
            normalized = key.casefold()
            if _is_literal_reader_path(child_path):
                yield child_path, None
                continue
            if normalized in excluded_subtrees:
                yield child_path, None
                continue
            if normalized in excluded_keys or normalized.endswith(
                ("_url", "_sha256", "_digest")
            ):
                yield child_path, None
                continue
            yield from _iter_reader_copy(
                child,
                path=child_path,
                excluded_subtrees=excluded_subtrees,
                excluded_keys=excluded_keys,
            )
        return
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for index, child in enumerate(value):
            yield from _iter_reader_copy(
                child,
                path=_json_path(path, index),
                excluded_subtrees=excluded_subtrees,
                excluded_keys=excluded_keys,
            )


def lint_reader_copy_tree(
    document: Any,
    *,
    max_issues: int = 500,
    excluded_subtrees: frozenset[str] = DEFAULT_EXCLUDED_READER_SUBTREES,
    excluded_keys: frozenset[str] = DEFAULT_EXCLUDED_READER_KEYS,
) -> RussianCopyLintReport:
    """Lint registered JSON-like reader copy while classifying immutable evidence.

    Raw model answers, citations, evidence and URLs are skipped by default.  A
    caller that publishes one of those fields as authored copy can pass narrower
    exclusion sets and audit it explicitly.
    """

    if max_issues < 1:
        raise ValueError("max_issues must be at least 1")

    all_issues: list[RussianCopyIssue] = []
    checked_characters = 0
    checked_fields = 0
    blocking_issue_count = 0
    field_omitted_issue_count = 0
    skipped_paths: list[str] = []
    for path, text in _iter_reader_copy(
        document,
        path="$",
        excluded_subtrees=excluded_subtrees,
        excluded_keys=excluded_keys,
    ):
        if text is None:
            skipped_paths.append(path)
            continue
        checked_fields += 1
        checked_characters += len(text)
        field_report = lint_russian_copy(text, path=path, max_issues=max_issues)
        blocking_issue_count += field_report.blocking_count
        all_issues.extend(field_report.issues)
        # ``field_report`` can bound findings for a pathological single field.
        # Preserve that count rather than pretending the report is exhaustive.
        field_omitted_issue_count += field_report.omitted_issue_count

    all_issues.sort(key=lambda issue: (issue.path, issue.start, issue.end, issue.code))
    omitted = field_omitted_issue_count + max(0, len(all_issues) - max_issues)
    return RussianCopyLintReport(
        policy_version=LIVE_RUSSIAN_POLICY_VERSION,
        policy_sha256=LIVE_RUSSIAN_POLICY_SHA256,
        checked_characters=checked_characters,
        checked_fields=checked_fields,
        issues=tuple(all_issues[:max_issues]),
        blocking_issue_count=blocking_issue_count,
        omitted_issue_count=omitted,
        skipped_paths=tuple(skipped_paths),
    )


def issue_with_path(issue: RussianCopyIssue, path: str) -> RussianCopyIssue:
    """Attach a manifest path to a finding produced before tree registration."""

    return replace(issue, path=path)


def has_blocking_copy_issues(value: str | RussianCopyLintReport) -> bool:
    """Small integration helper for preflight and cache acceptance gates."""

    report = lint_russian_copy(value) if isinstance(value, str) else value
    if not isinstance(report, RussianCopyLintReport):
        raise TypeError("value must be text or a RussianCopyLintReport")
    return report.blocking
