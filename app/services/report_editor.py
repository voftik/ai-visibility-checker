"""Fact-preserving Russian editorial pass for reader-facing report prose.

The editor is intentionally downstream from semantic analysis.  It may improve
wording, rhythm and typography, but it cannot change measurements, entities,
URLs, evidence or the set of claims.  Long text is divided into lossless core
units with overlap context; the number of units is not capped.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Awaitable, Callable, Iterable

from app.services.long_response import split_lossless_text, text_sha256


REPORT_EDITOR_POLICY_VERSION = "aiv-ru-editorial-policy-v2"
REPORT_EDITOR_HARNESS_VERSION = "aiv-report-editor-lossless-v2"
REPORT_EDITOR_UNIT_CHARS = 12_000  # working window, never a corpus limit
REPORT_EDITOR_CONTEXT_CHARS = 1_200
REPORT_EDITOR_MAX_REVISIONS = 1

_URL_RE = re.compile(r"https?://[^\s<>()\[\]{}]+", re.IGNORECASE)
_NUMBER_RE = re.compile(
    r"(?<![\w])[-+]?\d+(?:[.,]\d+)?(?:\s*(?:%|п\.\s*п\.|балл(?:а|ов)?|"
    r"страниц(?:а|ы)?|ответ(?:а|ов)?|из\s+\d+))?",
    re.IGNORECASE,
)
_LATIN_TERM_RE = re.compile(r"(?<![\w])[A-Za-z][A-Za-z0-9.+/_-]{1,}(?![\w])")
_SENTENCE_RE = re.compile(r"[^\n.!?]+(?:[.!?]+|\n+|$)", re.UNICODE)
_INTERNAL_META_RE = re.compile(
    r"\b(?:artifact(?:_key)?|prompt_version|source_sha256|claim_id|fact_ref|"
    r"enum|JSON[- ]?ключ|пайплайн|code-owned|lineage)\b",
    re.IGNORECASE,
)
_FORBIDDEN_STYLE_RE = re.compile(
    r"\b(?:важно отметить|в современном мире|комплексный подход|"
    r"следует отметить|данный (?:раздел|показатель|отч[её]т)|"
    r"как мы видим|как видно (?:из|на)|само собой разумеется|"
    r"(?:это |совершенно )?очевидно|ниже представлен[аоы]?|"
    r"в этом разделе (?:мы )?(?:рассмотрим|покажем|разбер[её]м)|"
    r"(?:эта|данная) (?:диаграмма|таблица|схема) показывает|"
    r"три независимых среза|ось начинается с нуля|"
    r"целевая марка оста[её]тся на диаграмме)\b",
    re.IGNORECASE,
)
_PASSIVE_STYLE_RE = re.compile(
    r"\b(?:анализиру|рассматрива|осуществля|производ|выполня|формиру|"
    r"определя|устанавлива|предоставля|обеспечива)(?:ется|ются)\b|"
    r"\b(?:был|была|было|были|будет|будут)\s+"
    r"(?:провед[её]н|принят|подготовлен|реализован|сформирован|"
    r"определ[её]н|установлен|предоставлен|загружен|заверш[её]н)"
    r"[а-яё-]*\b",
    re.IGNORECASE,
)
_SLOGAN_STYLE_RE = re.compile(
    r"\bэто\s+не\s+просто\b.{0,120}\b(?:а|это)\b|"
    r"\b(?:скорость|данные|доступ|видимость|узнавание)\s+(?:есть|полное)"
    r"[.!?]\s+(?:над[её]жности|выводов|понимания|рекомендаций)\s+(?:нет|"
    r"недостаточно)|"
    r"\b(?:быстро|над[её]жно|удобно|эффективно|качественно)\s*[.!]\s*"
    r"(?:быстро|над[её]жно|удобно|эффективно|качественно)\s*[.!]\s*"
    r"(?:быстро|над[её]жно|удобно|эффективно|качественно)\b",
    re.IGNORECASE | re.DOTALL,
)
_MECHANICAL_TRIAD_RE = re.compile(
    r"\b(?:быстр\w*|над[её]жн\w*|удобн\w*|эффективн\w*|качественн\w*|"
    r"инновационн\w*|анализируем|оптимизируем|масштабируем)\s*,\s*"
    r"(?:быстр\w*|над[её]жн\w*|удобн\w*|эффективн\w*|качественн\w*|"
    r"инновационн\w*|анализируем|оптимизируем|масштабируем)\s*,\s*"
    r"(?:быстр\w*|над[её]жн\w*|удобн\w*|эффективн\w*|качественн\w*|"
    r"инновационн\w*|анализируем|оптимизируем|масштабируем)\b",
    re.IGNORECASE,
)
_ORPHAN_NUMBER_RE = re.compile(
    r"(?:^|[.!?]\s+)\s*\d+(?:[.,]\d+)?%\s+"
    r"(?:уже\s+)?(?:используют|знают|выбирают|рекомендуют|назвали|"
    r"доступны|готовы)\b|"
    r"(?:^|\n)\s*\d+(?:[.,]\d+)?(?:\s*%|\s+балл(?:а|ов)?)?\s*"
    r"(?:[.!?]|$)",
    re.IGNORECASE,
)


EDITORIAL_POLICY = """
Ты выпускающий редактор русскоязычного аналитического отчёта. Улучши только
формулировку переданного core_text. Не добавляй, не удаляй, не усиливай и не
ослабляй факты. Сохрани действующее лицо, предмет каждого наблюдения, числа,
знаменатели, единицы, названия, URL, причинность и границы выборки.

Пиши для руководителя маркетинга или продукта. Заголовок сообщает законченный
вывод. Сначала вывод, затем доказательство и действие. Отделяй наблюдение от
интерпретации. Убирай канцелярит, пассив, рекламные эпитеты, синтетические
надзаголовки, служебное повествование о структуре отчёта и фразы капитана
очевидность. Не используй ложную формулу «это не X, а Y» и механические тройки.

Живой русский здесь означает конкретные проверяемые требования:
- в каждом действии назови деятеля или механизм, если он известен из исходника;
- у каждого числа назови носителя: что именно измерили и на какой выборке;
- пиши активным залогом; не прячь деятеля за «анализируется», «выполняется»,
  «было принято» и другими пассивными формами;
- не пиши лозунгами, рублеными антитезами и тремя симметричными словами ради
  ритма; сохраняй реальные три факта, но свяжи их естественной фразой;
- не объясняй устройство текста, таблицы или диаграммы и не сообщай читателю
  очевидное; сразу назови предметный вывод;
- никогда не используй длинное тире U+2014. Перестрой фразу через точку,
  запятую, двоеточие или глагол. Короткое тире в диапазоне 10–15 не меняй.

Если деятель или носитель числа не следует из исходного текста, не придумывай
его. Сохрани точную безличную формулировку и укажи ограничение в receipt.
Используй «ёлочки», десятичную запятую и неразрывные пробелы с единицами.
Латинские бренды и названия моделей не склоняй.

Верни только JSON по схеме. edited_text относится только к core_text, overlap
нужен для понимания границы. Для каждого source_claim верни receipt в том же
порядке и укажи точный target_excerpt из edited_text, который сохраняет смысл.
new_claims должен быть пустым.
""".strip()


EDITOR_UNIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "source_unit_id": {"type": "string"},
        "source_sha256": {"type": "string"},
        "edited_text": {"type": "string"},
        "claim_receipts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "claim_sha256": {"type": "string"},
                    "preserved": {"type": "boolean"},
                    "target_excerpt": {"type": "string"},
                    "note": {"type": "string"},
                },
                "required": [
                    "claim_sha256", "preserved", "target_excerpt", "note"
                ],
            },
        },
        "new_claims": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "source_unit_id", "source_sha256", "edited_text",
        "claim_receipts", "new_claims",
    ],
}


EDITOR_CRITIC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "verdict": {"type": "string", "enum": ["pass", "revise", "block"]},
        "issues": {"type": "array", "items": {"type": "string"}},
        "claim_checks": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "claim_sha256": {"type": "string"},
                    "meaning_preserved": {"type": "boolean"},
                    "actor_preserved": {"type": "boolean"},
                    "scope_preserved": {"type": "boolean"},
                    "numbers_preserved": {"type": "boolean"},
                    "actor_or_mechanism_explicit": {"type": "boolean"},
                    "number_carrier_explicit": {"type": "boolean"},
                    "active_voice": {"type": "boolean"},
                    "no_slogan_or_meta": {"type": "boolean"},
                    "no_mechanical_triad": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": [
                    "claim_sha256", "meaning_preserved", "actor_preserved",
                    "scope_preserved", "numbers_preserved",
                    "actor_or_mechanism_explicit", "number_carrier_explicit",
                    "active_voice", "no_slogan_or_meta",
                    "no_mechanical_triad", "reason",
                ],
            },
        },
        "new_claims": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["verdict", "issues", "claim_checks", "new_claims"],
}


EDITOR_CRITIC_POLICY = """
Ты независимый фактчекер редакторской правки. Сравни source_text и edited_text.
Проверь каждый claim_sha256: сохранились ли смысл, действующее лицо, граница
выборки, модальность, причинность, числа и единицы. Любое новое утверждение,
исчезнувшая оговорка, unknown превращённый в ноль, «может» превращённое в
«делает» или наблюдаемая связь, превращённая в причину, требует block. Стилевая
шероховатость без фактического искажения требует revise.

Отдельно проверь: у каждого известного действия назван деятель или механизм;
у каждого числа назван носитель; пассив не прячет деятеля; в edited_text нет
длинного тире U+2014, лозунга, фразы капитана очевидность, механической триады
или мета-повествования о разделе, таблице, диаграмме и процессе анализа.
Не требуй выдумать деятеля или носителя, которого нет в source_text.
Верни только JSON.
""".strip()


ModelCall = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class EditorialUnit:
    unit_id: str
    path: str
    index: int
    start_char: int
    end_char: int
    source_text: str
    context_text: str
    core_start_in_context: int
    core_end_in_context: int
    source_sha256: str
    claims: tuple[dict[str, str], ...]
    protected_terms: tuple[str, ...]

    def payload(self) -> dict[str, Any]:
        return {
            "policy_version": REPORT_EDITOR_POLICY_VERSION,
            "source_unit_id": self.unit_id,
            "source_sha256": self.source_sha256,
            "path": self.path,
            "core_text": self.source_text,
            "overlap_context": self.context_text,
            "core_start_in_context": self.core_start_in_context,
            "core_end_in_context": self.core_end_in_context,
            "source_claims": [copy.deepcopy(item) for item in self.claims],
            "protected_terms": list(self.protected_terms),
        }


@dataclass(frozen=True)
class EditorialAudit:
    version: str
    policy_version: str
    policy_sha256: str
    source_report_sha256: str
    result_report_sha256: str
    unit_count: int
    processed_unit_count: int
    changed_paths: tuple[str, ...]
    fallback_units: tuple[dict[str, str], ...]
    critic_verdicts: tuple[dict[str, str], ...]
    coverage_complete: bool

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["changed_paths"] = list(self.changed_paths)
        value["fallback_units"] = [dict(item) for item in self.fallback_units]
        value["critic_verdicts"] = [dict(item) for item in self.critic_verdicts]
        value["audit_sha256"] = _stable_sha256(value)
        return value


def _stable_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _pointer_get(document: Any, path: str) -> Any:
    value = document
    for raw in path.split("/")[1:]:
        key = raw.replace("~1", "/").replace("~0", "~")
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def _pointer_set(document: Any, path: str, replacement: str) -> None:
    parts = path.split("/")[1:]
    value = document
    for raw in parts[:-1]:
        key = raw.replace("~1", "/").replace("~0", "~")
        value = value[int(key)] if isinstance(value, list) else value[key]
    final = parts[-1].replace("~1", "/").replace("~0", "~")
    if isinstance(value, list):
        value[int(final)] = replacement
    else:
        value[final] = replacement


def _existing_nonempty_string_paths(
    document: dict[str, Any],
    candidates: Iterable[str],
) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for raw_path in candidates:
        path = str(raw_path)
        if path in seen:
            continue
        seen.add(path)
        try:
            value = _pointer_get(document, path)
        except (IndexError, KeyError, TypeError, ValueError):
            continue
        if isinstance(value, str) and value.strip():
            paths.append(path)
    return paths


def reader_narrative_paths(report: dict[str, Any]) -> list[str]:
    """Return only prose fields; raw evidence and canonical limits stay exact."""

    paths = ["/headline", "/verdict", "/executive_summary"]
    for index, section in enumerate(report.get("sections") or []):
        if isinstance(section, dict):
            paths.extend((f"/sections/{index}/heading", f"/sections/{index}/body"))
    for index, action in enumerate(report.get("actions") or []):
        if not isinstance(action, dict):
            continue
        # evidence is an immutable factual anchor and is deliberately excluded.
        paths.extend(
            f"/actions/{index}/{field}" for field in ("title", "why", "step")
        )
    return _existing_nonempty_string_paths(report, paths)


def technical_review_narrative_paths(review: dict[str, Any]) -> list[str]:
    """Select technical-review prose while excluding evidence and enums."""

    paths = ["/overall_conclusion", "/render_conclusion"]
    for index, finding in enumerate(review.get("findings") or []):
        if not isinstance(finding, dict):
            continue
        # severity is a schema enum; evidence is an immutable factual anchor.
        paths.extend(
            f"/findings/{index}/{field}"
            for field in ("title", "business_effect", "action")
        )
    for index, limitation in enumerate(review.get("limitations") or []):
        if isinstance(limitation, str):
            paths.append(f"/limitations/{index}")
    return _existing_nonempty_string_paths(review, paths)


def _explicit_narrative_paths(
    document: dict[str, Any],
    prose_paths: Iterable[str],
) -> list[str]:
    """Validate caller-owned JSON Pointers before any model sees their text."""

    paths: list[str] = []
    seen: set[str] = set()
    for raw_path in prose_paths:
        if not isinstance(raw_path, str) or not raw_path.startswith("/"):
            raise ValueError("prose_paths must contain absolute JSON Pointers")
        if raw_path in seen:
            continue
        seen.add(raw_path)
        try:
            value = _pointer_get(document, raw_path)
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"prose path does not resolve: {raw_path}"
            ) from exc
        if not isinstance(value, str):
            raise ValueError(f"prose path is not a string: {raw_path}")
        if value.strip():
            paths.append(raw_path)
    return paths


def _claims(text: str) -> tuple[dict[str, str], ...]:
    parts = [match.group(0).strip() for match in _SENTENCE_RE.finditer(text)]
    if not parts and text.strip():
        parts = [text.strip()]
    return tuple(
        {
            "claim_sha256": text_sha256(part),
            "source_excerpt": part,
        }
        for part in parts
        if part
    )


def _protected_terms(text: str, extra: Iterable[str]) -> tuple[str, ...]:
    without_urls = _URL_RE.sub(" ", text)
    terms = [match.group(0) for match in _LATIN_TERM_RE.finditer(without_urls)]
    terms.extend(str(item).strip() for item in extra if str(item).strip())
    # Long generic words supplied as an alias may occur in normal prose.  Exact
    # preservation remains safe: the editor may move, but not rename, entities.
    return tuple(dict.fromkeys(terms))


def build_editorial_units(
    report: dict[str, Any],
    *,
    prose_paths: Iterable[str] | None = None,
    protected_terms: Iterable[str] = (),
    target_chars: int = REPORT_EDITOR_UNIT_CHARS,
) -> tuple[list[EditorialUnit], dict[str, Any]]:
    units: list[EditorialUnit] = []
    path_manifests: list[dict[str, Any]] = []
    selected_paths = (
        reader_narrative_paths(report)
        if prose_paths is None
        else _explicit_narrative_paths(report, prose_paths)
    )
    for path in selected_paths:
        text = str(_pointer_get(report, path))
        document_id = "editor:" + text_sha256(path)[:16]
        text_units, manifest = split_lossless_text(
            text,
            document_id=document_id,
            target_chars=target_chars,
            context_overlap_chars=REPORT_EDITOR_CONTEXT_CHARS,
        )
        path_manifests.append({"path": path, "manifest": manifest.as_dict()})
        for item in text_units:
            units.append(
                EditorialUnit(
                    unit_id=item.unit_id,
                    path=path,
                    index=item.index,
                    start_char=item.start_char,
                    end_char=item.end_char,
                    source_text=item.text,
                    context_text=item.context_text,
                    core_start_in_context=item.core_start_in_context,
                    core_end_in_context=item.core_end_in_context,
                    source_sha256=item.sha256,
                    claims=_claims(item.text),
                    protected_terms=_protected_terms(item.text, protected_terms),
                )
            )
    manifest_core = {
        "version": REPORT_EDITOR_HARNESS_VERSION,
        "source_report_sha256": _stable_sha256(report),
        "unit_count": len(units),
        "unit_ids": [item.unit_id for item in units],
        "path_selection": (
            "reader_report_default" if prose_paths is None else "explicit_json_pointer"
        ),
        "prose_paths": selected_paths,
        "path_manifests": path_manifests,
        "coverage_complete": True,
    }
    return units, {**manifest_core, "manifest_sha256": _stable_sha256(manifest_core)}


def _normalized_numbers(text: str) -> Counter[str]:
    values: list[str] = []
    for match in _NUMBER_RE.finditer(text):
        token = re.sub(r"\s+", " ", match.group(0).strip().casefold())
        token = re.sub(r"(?<=\d),(?=\d)", ".", token)
        values.append(token)
    return Counter(values)


def _literal_counts(text: str, values: Iterable[str]) -> Counter[str]:
    folded = text.casefold()
    return Counter(
        {
            value: len(re.findall(re.escape(value.casefold()), folded))
            for value in values
        }
    )


def validate_editor_result(
    unit: EditorialUnit,
    result: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    if result.get("source_unit_id") != unit.unit_id:
        errors.append("source_unit_id_mismatch")
    if result.get("source_sha256") != unit.source_sha256:
        errors.append("source_sha256_mismatch")
    edited = result.get("edited_text")
    if not isinstance(edited, str) or (unit.source_text.strip() and not edited.strip()):
        errors.append("edited_text_missing")
        edited = ""
    if result.get("new_claims") not in ([], None):
        errors.append("new_claims_reported")
    receipts = result.get("claim_receipts")
    expected = [item["claim_sha256"] for item in unit.claims]
    actual = [
        str(item.get("claim_sha256") or "")
        for item in receipts or []
        if isinstance(item, dict)
    ]
    if actual != expected:
        errors.append("claim_receipt_coverage_mismatch")
    for receipt in receipts or []:
        if not isinstance(receipt, dict):
            errors.append("claim_receipt_invalid")
            continue
        excerpt = str(receipt.get("target_excerpt") or "")
        if receipt.get("preserved") is not True or not excerpt or excerpt not in edited:
            errors.append("claim_not_visibly_preserved")
    if Counter(_URL_RE.findall(unit.source_text)) != Counter(_URL_RE.findall(edited)):
        errors.append("url_set_changed")
    if _normalized_numbers(unit.source_text) != _normalized_numbers(edited):
        errors.append("number_or_unit_set_changed")
    if _literal_counts(unit.source_text, unit.protected_terms) != _literal_counts(
        edited, unit.protected_terms
    ):
        errors.append("protected_name_set_changed")
    if _INTERNAL_META_RE.search(edited) and not _INTERNAL_META_RE.search(
        unit.source_text
    ):
        errors.append("internal_pipeline_meta_added")
    if _FORBIDDEN_STYLE_RE.search(edited):
        errors.append("forbidden_editorial_boilerplate")
    if _PASSIVE_STYLE_RE.search(edited):
        errors.append("avoidable_passive_voice")
    if _SLOGAN_STYLE_RE.search(edited):
        errors.append("slogan_or_hollow_antithesis")
    if _MECHANICAL_TRIAD_RE.search(edited):
        errors.append("mechanical_triad")
    if _ORPHAN_NUMBER_RE.search(edited):
        errors.append("number_carrier_missing")
    if "—" in edited:
        errors.append("long_dash_forbidden")
    return list(dict.fromkeys(errors))


def validate_critic_result(
    unit: EditorialUnit,
    result: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    checks = result.get("claim_checks")
    expected = [item["claim_sha256"] for item in unit.claims]
    actual = [
        str(item.get("claim_sha256") or "")
        for item in checks or []
        if isinstance(item, dict)
    ]
    if actual != expected:
        errors.append("critic_claim_coverage_mismatch")
    for item in checks or []:
        if not isinstance(item, dict) or not all(
            item.get(key) is True
            for key in (
                "meaning_preserved", "actor_preserved", "scope_preserved",
                "numbers_preserved",
            )
        ):
            errors.append("critic_found_semantic_drift")
        if not isinstance(item, dict) or not all(
            item.get(key) is True
            for key in (
                "actor_or_mechanism_explicit", "number_carrier_explicit",
                "active_voice", "no_slogan_or_meta",
                "no_mechanical_triad",
            )
        ):
            errors.append("critic_found_live_russian_defect")
    if result.get("new_claims") not in ([], None):
        errors.append("critic_found_new_claims")
    if any(str(issue).strip() for issue in result.get("issues") or []):
        errors.append("critic_reported_issues")
    if result.get("verdict") != "pass":
        errors.append("critic_did_not_pass")
    return list(dict.fromkeys(errors))


async def edit_report(
    report: dict[str, Any],
    *,
    editor_call: ModelCall,
    critic_call: ModelCall,
    arbiter_call: ModelCall | None = None,
    prose_paths: Iterable[str] | None = None,
    protected_terms: Iterable[str] = (),
    concurrency: int = 4,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Edit selected prose with one bounded revision and safe fallback."""

    source = copy.deepcopy(report)
    units, manifest = build_editorial_units(
        source,
        prose_paths=prose_paths,
        protected_terms=protected_terms,
    )
    semaphore = asyncio.Semaphore(max(1, int(concurrency)))

    async def process(unit: EditorialUnit) -> tuple[str, str, dict[str, str]]:
        errors: list[str] = []
        candidate: dict[str, Any] | None = None
        critic: dict[str, Any] | None = None
        async with semaphore:
            for attempt in range(REPORT_EDITOR_MAX_REVISIONS + 1):
                payload = unit.payload()
                payload["editorial_policy"] = EDITORIAL_POLICY
                payload["response_schema"] = EDITOR_UNIT_SCHEMA
                payload["attempt"] = attempt
                if candidate is not None:
                    payload["rejected_candidate"] = candidate
                    payload["validation_errors_to_fix"] = errors
                    payload["critic_review_to_fix"] = critic
                try:
                    candidate = await editor_call(payload)
                except Exception as exc:  # transport is audited by the caller
                    errors = [f"editor_call_failed:{type(exc).__name__}"]
                    break
                errors = validate_editor_result(unit, candidate)
                if errors:
                    continue
                critic_payload = {
                    "policy": EDITOR_CRITIC_POLICY,
                    "source_unit_id": unit.unit_id,
                    "source_text": unit.source_text,
                    "source_claims": list(unit.claims),
                    "edited_text": candidate["edited_text"],
                    "response_schema": EDITOR_CRITIC_SCHEMA,
                }
                try:
                    critic = await critic_call(critic_payload)
                except Exception as exc:
                    errors = [f"critic_call_failed:{type(exc).__name__}"]
                    continue
                errors = validate_critic_result(unit, critic)
                if not errors:
                    return (
                        unit.unit_id,
                        str(candidate["edited_text"]),
                        {"unit_id": unit.unit_id, "verdict": "pass"},
                    )

            if arbiter_call is not None and candidate is not None:
                arbiter_payload = unit.payload()
                arbiter_payload.update(
                    {
                        "editorial_policy": EDITORIAL_POLICY,
                        "candidate": candidate,
                        "critic": critic,
                        "errors": errors,
                        "instruction": (
                            "Как старший выпускающий редактор верни финальную "
                            "фактосохраняющую редакцию по EDITOR_UNIT_SCHEMA."
                        ),
                        "response_schema": EDITOR_UNIT_SCHEMA,
                    }
                )
                try:
                    arbitrated = await arbiter_call(arbiter_payload)
                    arbiter_errors = validate_editor_result(unit, arbitrated)
                    if not arbiter_errors:
                        final_critic = await critic_call(
                            {
                                "policy": EDITOR_CRITIC_POLICY,
                                "source_unit_id": unit.unit_id,
                                "source_text": unit.source_text,
                                "source_claims": list(unit.claims),
                                "edited_text": arbitrated["edited_text"],
                                "response_schema": EDITOR_CRITIC_SCHEMA,
                            }
                        )
                        if not validate_critic_result(unit, final_critic):
                            return (
                                unit.unit_id,
                                str(arbitrated["edited_text"]),
                                {"unit_id": unit.unit_id, "verdict": "arbiter_pass"},
                            )
                        errors = validate_critic_result(unit, final_critic)
                    else:
                        errors = arbiter_errors
                except Exception as exc:
                    errors = [f"arbiter_call_failed:{type(exc).__name__}"]

        return (
            unit.unit_id,
            unit.source_text,
            {
                "unit_id": unit.unit_id,
                "verdict": "fallback",
                "reason": ";".join(errors) or "editorial_validation_failed",
            },
        )

    outcomes = await asyncio.gather(*(process(unit) for unit in units))
    by_id = {unit_id: (text, audit) for unit_id, text, audit in outcomes}
    rebuilt: dict[str, list[tuple[int, str]]] = defaultdict(list)
    critic_rows: list[dict[str, str]] = []
    fallback_rows: list[dict[str, str]] = []
    for unit in units:
        edited, row = by_id[unit.unit_id]
        rebuilt[unit.path].append((unit.index, edited))
        critic_rows.append(row)
        if row.get("verdict") == "fallback":
            fallback_rows.append(row)

    result = copy.deepcopy(source)
    changed_paths: list[str] = []
    for path, parts in rebuilt.items():
        value = "".join(text for _index, text in sorted(parts))
        if value != _pointer_get(source, path):
            changed_paths.append(path)
        _pointer_set(result, path, value)
    # The product design requires one even weight/size in the report hero.
    # Emphasis metadata is presentation-only and must never survive a rewrite.
    if "headline_emphasis" in result:
        result["headline_emphasis"] = []

    processed_ids = [row[0] for row in outcomes]
    coverage_complete = (
        len(processed_ids) == len(units)
        and len(set(processed_ids)) == len(units)
        and set(processed_ids) == {unit.unit_id for unit in units}
    )
    if not coverage_complete:
        result = source
        fallback_rows.append(
            {"unit_id": "*", "verdict": "fallback", "reason": "coverage_incomplete"}
        )
    audit = EditorialAudit(
        version=REPORT_EDITOR_HARNESS_VERSION,
        policy_version=REPORT_EDITOR_POLICY_VERSION,
        policy_sha256=text_sha256(EDITORIAL_POLICY + "\n" + EDITOR_CRITIC_POLICY),
        source_report_sha256=manifest["source_report_sha256"],
        result_report_sha256=_stable_sha256(result),
        unit_count=len(units),
        processed_unit_count=len(processed_ids),
        changed_paths=tuple(sorted(changed_paths)),
        fallback_units=tuple(copy.deepcopy(fallback_rows)),
        critic_verdicts=tuple(copy.deepcopy(critic_rows)),
        coverage_complete=coverage_complete,
    ).as_dict()
    audit["source_manifest"] = manifest
    return result, audit
