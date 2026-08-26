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
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import asdict, dataclass
from typing import Any

from app.services.live_russian_policy import (
    LIVE_RUSSIAN_POLICY_MANIFEST,
    build_live_russian_policy_prompt,
    lint_reader_copy_tree,
    lint_russian_copy,
    trusted_live_russian_policy_manifest,
)
from app.services.long_response import (
    LOSSLESS_PARTITION_VERSION,
    split_lossless_text,
    text_sha256,
)

REPORT_EDITOR_POLICY_VERSION = "aiv-ru-editorial-policy-v4"
REPORT_EDITOR_HARNESS_VERSION = "aiv-report-editor-lossless-v6"
REPORT_EDITOR_BOUNDARY_VERSION = "aiv-editor-code-owned-boundary-v1"
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


EDITORIAL_POLICY = (
    """
Ты выпускающий редактор русскоязычного аналитического отчёта. Улучши только
формулировку переданного core_text. Не добавляй, не удаляй, не усиливай и не
ослабляй факты. Сохрани действующее лицо, предмет каждого наблюдения, числа,
знаменатели, единицы, названия, URL, причинность и границы выборки.

Разрешай конфликты в таком порядке: читатель должен правильно понять мысль;
сила утверждения должна точно совпасть с исходником; фраза должна звучать так,
как её сказал бы человек; формальные стилевые правила идут после этого. Если
живая формулировка искажает смысл, сохрани точную исходную формулировку.

Пиши для руководителя маркетинга или продукта. В аналитике субъектом служат
измеренная система, сайт, модель, пользователь или команда, а не безымянный
«анализ». Заголовок содержательного блока сообщает законченную мысль и понятен
без иллюстрации; короткие подписи, названия таблиц и кнопок могут оставаться
назывными. Сначала вывод, затем доказательство и действие. Отделяй наблюдение
от интерпретации. Плохой результат называй прямо, не прячь его в вводке.

Убирай канцелярит, отглагольные конструкции, пустые слова-контейнеры,
рекламные эпитеты, синтетические надзаголовки, служебное повествование о
структуре отчёта и фразы капитана очевидность. Ставь глагол вместо
«осуществления», «проведения» и «обеспечения», но не ломай нормальные термины
вроде «бронирование» или «обучение». Не заменяй конкретное название общим
«решением», «направлением», «контуром» или «инициативой». Не используй ложную
формулу «это не X, а Y», парцелляцию ради драматизма и механические тройки.
Не добавляй вступление, пересказ задачи, «таким образом» или финальную
любезность. Не хеджируй ради вежливости, но обязательно сохрани исходные
«может», «вероятно», измеренную долю и любое настоящее ограничение.

Живой русский здесь означает конкретные проверяемые требования:
- в каждом действии назови деятеля или механизм, если он известен из исходника;
- у каждого числа назови носителя: что именно измерили и на какой выборке;
- пиши активным залогом; не прячь деятеля за «анализируется», «выполняется»,
  «было принято» и другими пассивными формами;
- не пиши лозунгами, рублеными антитезами и тремя симметричными словами ради
  ритма; сохраняй реальные три факта, но свяжи их естественной фразой;
- не объясняй устройство текста, таблицы или диаграммы и не сообщай читателю
  очевидное; сразу назови предметный вывод;
- называй конкретный сервис, страницу, модель, продукт или действие, если они
  уже есть в исходнике; никогда не достраивай правдоподобную конкретику;
- чередуй длину предложений естественно, не делай одинаковые абзацы и не ставь
  подряд три рубленые фразы; структура должна следовать материалу, а не шаблону;
- латинское название не бери в кавычки, не склоняй и не меняй его регистр;
- используй «ёлочки», десятичную запятую и неразрывный пробел перед единицей,
  если это не меняет защищённые числа и буквальные названия;
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
    + "\n\n"
    + build_live_russian_policy_prompt(
        context="report",
        preserve_facts=True,
    )
)


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
                "required": ["claim_sha256", "preserved", "target_excerpt", "note"],
            },
        },
        "new_claims": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "source_unit_id",
        "source_sha256",
        "edited_text",
        "claim_receipts",
        "new_claims",
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
                    "claim_sha256",
                    "meaning_preserved",
                    "actor_preserved",
                    "scope_preserved",
                    "numbers_preserved",
                    "actor_or_mechanism_explicit",
                    "number_carrier_explicit",
                    "active_voice",
                    "no_slogan_or_meta",
                    "no_mechanical_triad",
                    "reason",
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
Проверь, что заголовок передаёт законченную мысль, канцелярит и пустые
слова-контейнеры заменены конкретным глаголом, латинские названия не склонены,
мера уверенности не усилилась и не ослабла, а ритм не превратился в три
одинаковые рубленые фразы. Не требуй удалить подтверждённый факт только ради
ритма или «уникальности».
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
    editable_text: str
    code_owned_prefix: str
    code_owned_suffix: str
    partition_sha256: str
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
            "core_text": self.editable_text,
            "overlap_context": self.context_text,
            "core_start_in_context": self.core_start_in_context,
            "core_end_in_context": self.core_end_in_context,
            "boundary_contract": {
                "version": REPORT_EDITOR_BOUNDARY_VERSION,
                "instruction": (
                    "Верни только редакцию core_text без начальных и конечных "
                    "пробельных символов. Разделители между фрагментами "
                    "добавляет код."
                ),
                "partition_sha256": self.partition_sha256,
                "code_owned_prefix_chars": len(self.code_owned_prefix),
                "code_owned_prefix_sha256": text_sha256(self.code_owned_prefix),
                "code_owned_suffix_chars": len(self.code_owned_suffix),
                "code_owned_suffix_sha256": text_sha256(self.code_owned_suffix),
            },
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
    boundary_receipts: tuple[dict[str, Any], ...]
    path_coverage_complete: bool
    boundary_integrity_complete: bool
    quality_complete: bool
    coverage_complete: bool

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["changed_paths"] = list(self.changed_paths)
        value["fallback_units"] = [dict(item) for item in self.fallback_units]
        value["critic_verdicts"] = [dict(item) for item in self.critic_verdicts]
        value["boundary_receipts"] = [
            copy.deepcopy(item) for item in self.boundary_receipts
        ]
        return seal_editorial_audit(value)


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


def editorial_policy_sha256() -> str:
    return text_sha256(EDITORIAL_POLICY + "\n" + EDITOR_CRITIC_POLICY)


_ACCEPTED_EDITORIAL_VERDICTS = frozenset({"pass", "arbiter_pass"})


@dataclass(frozen=True, slots=True)
class EditorialContractManifest:
    """Code-owned identity of one retained editorial proof format."""

    harness_version: str
    policy_version: str
    policy_sha256: str
    boundary_version: str
    partition_version: str
    canonical_policy_items: tuple[tuple[str, str], ...]
    accepted_verdicts: tuple[str, ...] = ("arbiter_pass", "pass")

    def as_dict(self) -> dict[str, Any]:
        return {
            "harness_version": self.harness_version,
            "policy_version": self.policy_version,
            "policy_sha256": self.policy_sha256,
            "boundary_version": self.boundary_version,
            "partition_version": self.partition_version,
            "canonical_policy": dict(self.canonical_policy_items),
            "accepted_verdicts": list(self.accepted_verdicts),
        }


# This literal is deliberately independent from the mutable current constants.
# Keep it (and the referenced policy bytes) when introducing the next editor.
_EDITORIAL_CONTRACT_V6 = EditorialContractManifest(
    harness_version="aiv-report-editor-lossless-v6",
    policy_version="aiv-ru-editorial-policy-v4",
    policy_sha256=("f1e7a11fda3a03e275bdc5b1ddd0c27e48f4d5f4868c959bf80ce53a12f0cffd"),
    boundary_version="aiv-editor-code-owned-boundary-v1",
    partition_version="aiv-lossless-partition-v2",
    canonical_policy_items=(
        ("language", "ru"),
        ("sha256", "0cd7bbc6cdb006331b3df3c414cc0cdb9bc9860dfa2706b098fe610778392d84"),
        ("snapshot", "app/policies/live_russian_ru.v2026-07-29.md"),
        ("source_date", "2026-07-29"),
        ("version", "live-russian-2026-07-29.1"),
    ),
)

CURRENT_EDITORIAL_CONTRACT_MANIFEST = EditorialContractManifest(
    harness_version=REPORT_EDITOR_HARNESS_VERSION,
    policy_version=REPORT_EDITOR_POLICY_VERSION,
    policy_sha256=editorial_policy_sha256(),
    boundary_version=REPORT_EDITOR_BOUNDARY_VERSION,
    partition_version=LOSSLESS_PARTITION_VERSION,
    canonical_policy_items=tuple(
        sorted(LIVE_RUSSIAN_POLICY_MANIFEST.as_dict().items())
    ),
)

# Historical reads resolve only against this code-owned archive.  The current
# descriptor is included so a deliberate version bump can publish immediately;
# old descriptors remain explicit literals and cannot be invented by database
# input, even when every outer JSON digest is coherently resealed.
TRUSTED_EDITORIAL_CONTRACT_MANIFESTS: tuple[EditorialContractManifest, ...] = tuple(
    dict.fromkeys(
        (
            _EDITORIAL_CONTRACT_V6,
            CURRENT_EDITORIAL_CONTRACT_MANIFEST,
        )
    )
)


def _editorial_quality_complete(audit: dict[str, Any]) -> bool:
    """Derive cache quality independently from lossless path coverage."""

    unit_count = audit.get("unit_count")
    processed_unit_count = audit.get("processed_unit_count")
    fallback_units = audit.get("fallback_units")
    critic_verdicts = audit.get("critic_verdicts")
    boundary_receipts = audit.get("boundary_receipts")
    source_manifest = audit.get("source_manifest")
    boundary_contract = (
        source_manifest.get("boundary_contract")
        if isinstance(source_manifest, dict)
        else None
    )
    if (
        audit.get("version") != REPORT_EDITOR_HARNESS_VERSION
        or audit.get("policy_version") != REPORT_EDITOR_POLICY_VERSION
        or audit.get("policy_sha256") != editorial_policy_sha256()
        or audit.get("path_coverage_complete") is not True
        or audit.get("boundary_integrity_complete") is not True
        or audit.get("coverage_complete") is not True
        or not isinstance(unit_count, int)
        or unit_count < 0
        or processed_unit_count != unit_count
        or not isinstance(fallback_units, list)
        or fallback_units
        or not isinstance(critic_verdicts, list)
        or len(critic_verdicts) != unit_count
        or not isinstance(boundary_receipts, list)
        or len(boundary_receipts) != unit_count
        or not isinstance(source_manifest, dict)
        or source_manifest.get("version") != REPORT_EDITOR_HARNESS_VERSION
        or source_manifest.get("policy_version") != REPORT_EDITOR_POLICY_VERSION
        or not isinstance(boundary_contract, dict)
        or boundary_contract.get("version") != REPORT_EDITOR_BOUNDARY_VERSION
        or source_manifest.get("unit_count") != unit_count
    ):
        return False
    unit_ids = [row.get("unit_id") for row in critic_verdicts if isinstance(row, dict)]
    if (
        len(unit_ids) != unit_count
        or any(not isinstance(unit_id, str) or not unit_id for unit_id in unit_ids)
        or len(set(unit_ids)) != unit_count
    ):
        return False
    if any(
        not isinstance(row, dict)
        or row.get("verdict") not in _ACCEPTED_EDITORIAL_VERDICTS
        for row in critic_verdicts
    ):
        return False
    boundary_unit_ids = [
        row.get("unit_id") for row in boundary_receipts if isinstance(row, dict)
    ]
    if boundary_unit_ids != unit_ids:
        return False
    if source_manifest.get("unit_ids") != unit_ids:
        return False
    semantic_fallback = audit.get("semantic_fallback")
    if semantic_fallback is not None and (
        not isinstance(semantic_fallback, dict)
        or semantic_fallback.get("used") is not False
    ):
        return False
    if audit.get("canonical_policy") != LIVE_RUSSIAN_POLICY_MANIFEST.as_dict():
        return False
    copy_lint = audit.get("reader_copy_lint")
    if (
        not isinstance(copy_lint, dict)
        or copy_lint.get("policy_version") != LIVE_RUSSIAN_POLICY_MANIFEST.version
        or copy_lint.get("policy_sha256") != LIVE_RUSSIAN_POLICY_MANIFEST.sha256
        or copy_lint.get("blocking") is not False
        or copy_lint.get("issues") != []
        or copy_lint.get("omitted_issue_count") != 0
    ):
        return False
    return True


def seal_editorial_audit(audit: dict[str, Any]) -> dict[str, Any]:
    """Bind every persisted audit field, including coverage and quality."""

    value = copy.deepcopy(audit)
    value.pop("audit_sha256", None)
    value.setdefault(
        "canonical_policy",
        LIVE_RUSSIAN_POLICY_MANIFEST.as_dict(),
    )
    value.setdefault(
        "path_coverage_complete",
        value.get("coverage_complete") is True,
    )
    value["quality_complete"] = _editorial_quality_complete(value)
    value["audit_sha256"] = _stable_sha256(value)
    return value


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


def _pointer_escape(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _all_string_leaf_paths(value: Any, path: str = "") -> list[str]:
    """List every schema string independently from its edit classification."""

    paths: list[str] = []
    if isinstance(value, str):
        if path:
            paths.append(path)
        return paths
    if isinstance(value, list):
        for index, item in enumerate(value):
            paths.extend(_all_string_leaf_paths(item, f"{path}/{index}"))
        return paths
    if isinstance(value, dict):
        for key, item in value.items():
            paths.extend(
                _all_string_leaf_paths(item, f"{path}/{_pointer_escape(str(key))}")
            )
    return paths


def reader_rendered_string_paths(report: dict[str, Any]) -> list[str]:
    """All final-report strings that can reach Markdown or the report reader."""

    return _all_string_leaf_paths(report)


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
        paths.extend(f"/actions/{index}/{field}" for field in ("title", "why", "step"))
    for index, limitation in enumerate(report.get("limitations") or []):
        if isinstance(limitation, str):
            paths.append(f"/limitations/{index}")
    return _existing_nonempty_string_paths(report, paths)


def reader_immutable_passthrough_paths(report: dict[str, Any]) -> list[str]:
    """Final-report labels and factual anchors that the editor cannot rewrite."""

    paths: list[str] = []
    for index, value in enumerate(report.get("headline_emphasis") or []):
        if isinstance(value, str):
            paths.append(f"/headline_emphasis/{index}")
    for index, action in enumerate(report.get("actions") or []):
        if isinstance(action, dict):
            paths.extend((f"/actions/{index}/priority", f"/actions/{index}/evidence"))
    return _existing_nonempty_string_paths(report, paths)


def technical_review_rendered_string_paths(review: dict[str, Any]) -> list[str]:
    """All technical-review strings displayed by the report reader."""

    return _all_string_leaf_paths(review)


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


def technical_review_immutable_passthrough_paths(
    review: dict[str, Any],
) -> list[str]:
    """Technical evidence and enum labels must survive byte-for-byte."""

    paths: list[str] = []
    for index, finding in enumerate(review.get("findings") or []):
        if isinstance(finding, dict):
            paths.extend((f"/findings/{index}/severity", f"/findings/{index}/evidence"))
    return _existing_nonempty_string_paths(review, paths)


def illustration_copy_rendered_string_paths(
    document: dict[str, Any],
) -> list[str]:
    """All strings rendered next to an illustration or used as image alt."""

    return _all_string_leaf_paths(document)


def illustration_copy_narrative_paths(
    document: dict[str, Any],
) -> list[str]:
    paths: list[str] = []
    for index, item in enumerate(document.get("illustrations") or []):
        if not isinstance(item, dict):
            continue
        paths.extend(
            f"/illustrations/{index}/{field}"
            for field in ("title", "caption", "alt_text")
        )
    return _existing_nonempty_string_paths(document, paths)


def illustration_copy_immutable_passthrough_paths(
    document: dict[str, Any],
) -> list[str]:
    """Bind concept role and evidence pointers without sending them to the editor."""

    paths: list[str] = []
    for index, item in enumerate(document.get("illustrations") or []):
        if not isinstance(item, dict):
            continue
        paths.extend(
            (
                f"/illustrations/{index}/role",
                f"/illustrations/{index}/core_claim",
            )
        )
        for path_index, value in enumerate(item.get("evidence_paths") or []):
            if isinstance(value, str):
                paths.append(f"/illustrations/{index}/evidence_paths/{path_index}")
    return _existing_nonempty_string_paths(document, paths)


def _require_exact_string_fields(
    value: Any,
    *,
    path: str,
    fields: set[str],
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"invalid editorial document shape at {path}")
    if not all(isinstance(value.get(field), str) for field in fields):
        raise ValueError(f"invalid editorial string field at {path}")
    return value


def _require_string_array(value: Any, *, path: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"invalid editorial string array at {path}")
    return value


def _validate_final_report_shape(document: dict[str, Any]) -> None:
    expected_keys = {
        "headline",
        "headline_emphasis",
        "verdict",
        "executive_summary",
        "sections",
        "actions",
        "limitations",
    }
    if set(document) != expected_keys:
        raise ValueError("unsupported editorial document shape")
    for field in ("headline", "verdict", "executive_summary"):
        if not isinstance(document.get(field), str):
            raise ValueError(f"invalid editorial string field at /{field}")
    _require_string_array(document.get("headline_emphasis"), path="/headline_emphasis")
    _require_string_array(document.get("limitations"), path="/limitations")

    sections = document.get("sections")
    if not isinstance(sections, list):
        raise ValueError("invalid editorial row array at /sections")
    for index, section in enumerate(sections):
        _require_exact_string_fields(
            section,
            path=f"/sections/{index}",
            fields={"heading", "body"},
        )

    actions = document.get("actions")
    if not isinstance(actions, list):
        raise ValueError("invalid editorial row array at /actions")
    action_fields = {"priority", "title", "why", "step", "evidence"}
    for index, action in enumerate(actions):
        row = _require_exact_string_fields(
            action,
            path=f"/actions/{index}",
            fields=action_fields,
        )
        if row["priority"] not in {"now", "next", "later"}:
            raise ValueError(f"invalid editorial enum at /actions/{index}/priority")


def _validate_technical_review_shape(document: dict[str, Any]) -> None:
    expected_keys = {
        "overall_conclusion",
        "render_conclusion",
        "findings",
        "limitations",
    }
    if set(document) != expected_keys:
        raise ValueError("unsupported editorial document shape")
    for field in ("overall_conclusion", "render_conclusion"):
        if not isinstance(document.get(field), str):
            raise ValueError(f"invalid editorial string field at /{field}")
    _require_string_array(document.get("limitations"), path="/limitations")

    findings = document.get("findings")
    if not isinstance(findings, list):
        raise ValueError("invalid editorial row array at /findings")
    finding_fields = {
        "severity",
        "title",
        "evidence",
        "business_effect",
        "action",
    }
    for index, finding in enumerate(findings):
        row = _require_exact_string_fields(
            finding,
            path=f"/findings/{index}",
            fields=finding_fields,
        )
        if row["severity"] not in {"critical", "important", "observation"}:
            raise ValueError(f"invalid editorial enum at /findings/{index}/severity")


def _validate_illustration_copy_shape(document: dict[str, Any]) -> None:
    if set(document) != {"illustrations"}:
        raise ValueError("unsupported editorial document shape")
    illustrations = document.get("illustrations")
    if not isinstance(illustrations, list):
        raise ValueError("invalid editorial row array at /illustrations")
    for index, item in enumerate(illustrations):
        if not isinstance(item, dict) or set(item) != {
            "role",
            "core_claim",
            "title",
            "caption",
            "alt_text",
            "evidence_paths",
        }:
            raise ValueError(
                f"invalid editorial document shape at /illustrations/{index}"
            )
        if not all(
            isinstance(item.get(field), str)
            for field in ("role", "core_claim", "title", "caption", "alt_text")
        ):
            raise ValueError(
                f"invalid editorial string field at /illustrations/{index}"
            )
        _require_string_array(
            item.get("evidence_paths"),
            path=f"/illustrations/{index}/evidence_paths",
        )


def _editorial_path_contract(
    document: dict[str, Any],
) -> tuple[str, list[str], list[str], list[str]]:
    """Return independent expected/editable/immutable path registries."""

    technical_keys = {
        "overall_conclusion",
        "render_conclusion",
        "findings",
        "limitations",
    }
    final_keys = {
        "headline",
        "headline_emphasis",
        "verdict",
        "executive_summary",
        "sections",
        "actions",
        "limitations",
    }
    if technical_keys.issubset(document):
        _validate_technical_review_shape(document)
        kind = "technical_review"
        expected = technical_review_rendered_string_paths(document)
        editable = technical_review_narrative_paths(document)
        immutable = technical_review_immutable_passthrough_paths(document)
    elif final_keys.issubset(document):
        _validate_final_report_shape(document)
        kind = "final_report"
        expected = reader_rendered_string_paths(document)
        editable = reader_narrative_paths(document)
        immutable = reader_immutable_passthrough_paths(document)
    elif set(document) == {"illustrations"}:
        _validate_illustration_copy_shape(document)
        kind = "illustration_copy"
        expected = illustration_copy_rendered_string_paths(document)
        editable = illustration_copy_narrative_paths(document)
        immutable = illustration_copy_immutable_passthrough_paths(document)
    else:
        raise ValueError("unsupported editorial document shape")

    # Empty schema strings have no model unit, but are still covered and bound.
    immutable_set = set(immutable)
    for path in expected:
        if path not in editable and _pointer_get(document, path) == "":
            immutable_set.add(path)
    return (
        kind,
        expected,
        editable,
        [path for path in expected if path in immutable_set],
    )


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
            raise ValueError(f"prose path does not resolve: {raw_path}") from exc
        if not isinstance(value, str):
            raise ValueError(f"prose path is not a string: {raw_path}")
        if value.strip():
            paths.append(raw_path)
    return paths


def _split_code_owned_edge_whitespace(text: str) -> tuple[str, str, str]:
    """Keep exact unit-edge whitespace outside the model-owned text."""

    start = 0
    while start < len(text) and text[start].isspace():
        start += 1
    if start == len(text):
        return text, "", ""
    end = len(text)
    while end > start and text[end - 1].isspace():
        end -= 1
    return text[:start], text[start:end], text[end:]


def _boundary_manifest_record(unit: EditorialUnit) -> dict[str, Any]:
    return {
        "unit_id": unit.unit_id,
        "path": unit.path,
        "index": unit.index,
        "partition_sha256": unit.partition_sha256,
        "editable_source_sha256": unit.source_sha256,
        "code_owned_prefix_chars": len(unit.code_owned_prefix),
        "code_owned_prefix_utf8_bytes": len(unit.code_owned_prefix.encode("utf-8")),
        "code_owned_prefix_sha256": text_sha256(unit.code_owned_prefix),
        "code_owned_suffix_chars": len(unit.code_owned_suffix),
        "code_owned_suffix_utf8_bytes": len(unit.code_owned_suffix.encode("utf-8")),
        "code_owned_suffix_sha256": text_sha256(unit.code_owned_suffix),
    }


def _boundary_receipt(unit: EditorialUnit, edited_text: str) -> dict[str, Any]:
    if unit.source_text != (
        unit.code_owned_prefix + unit.editable_text + unit.code_owned_suffix
    ):
        raise ValueError("editorial source boundary contract is inconsistent")
    if text_sha256(unit.source_text) != unit.partition_sha256:
        raise ValueError("editorial partition digest is inconsistent")
    if text_sha256(unit.editable_text) != unit.source_sha256:
        raise ValueError("editorial editable-core digest is inconsistent")
    if edited_text != edited_text.strip():
        raise ValueError("editor returned code-owned edge whitespace")
    assembled = unit.code_owned_prefix + edited_text + unit.code_owned_suffix
    return {
        "version": REPORT_EDITOR_BOUNDARY_VERSION,
        "unit_id": unit.unit_id,
        "path": unit.path,
        "index": unit.index,
        "partition_sha256": unit.partition_sha256,
        "editable_result_chars": len(edited_text),
        "editable_result_utf8_bytes": len(edited_text.encode("utf-8")),
        "editable_result_sha256": text_sha256(edited_text),
        "code_owned_prefix_chars": len(unit.code_owned_prefix),
        "code_owned_prefix_sha256": text_sha256(unit.code_owned_prefix),
        "code_owned_suffix_chars": len(unit.code_owned_suffix),
        "code_owned_suffix_sha256": text_sha256(unit.code_owned_suffix),
        "assembled_chars": len(assembled),
        "assembled_utf8_bytes": len(assembled.encode("utf-8")),
        "assembled_sha256": text_sha256(assembled),
    }


def _boundary_assembly_is_valid(
    units: Iterable[EditorialUnit],
    result: dict[str, Any],
    receipts: Any,
) -> bool:
    """Re-slice a cached result and prove every code-owned edge is exact."""

    ordered_units = list(units)
    if not isinstance(receipts, list) or len(receipts) != len(ordered_units):
        return False
    cursors: dict[str, int] = defaultdict(int)
    touched_paths: set[str] = set()
    try:
        for unit, receipt in zip(ordered_units, receipts, strict=True):
            if not isinstance(receipt, dict):
                return False
            value = _pointer_get(result, unit.path)
            if not isinstance(value, str):
                return False
            cursor = cursors[unit.path]
            prefix_end = cursor + len(unit.code_owned_prefix)
            if value[cursor:prefix_end] != unit.code_owned_prefix:
                return False
            edited_chars = receipt.get("editable_result_chars")
            if (
                isinstance(edited_chars, bool)
                or not isinstance(edited_chars, int)
                or edited_chars < 0
            ):
                return False
            edited_end = prefix_end + edited_chars
            edited_text = value[prefix_end:edited_end]
            suffix_end = edited_end + len(unit.code_owned_suffix)
            if value[edited_end:suffix_end] != unit.code_owned_suffix:
                return False
            if receipt != _boundary_receipt(unit, edited_text):
                return False
            cursors[unit.path] = suffix_end
            touched_paths.add(unit.path)
        for path in touched_paths:
            value = _pointer_get(result, path)
            if cursors[path] != len(value):
                return False
    except (IndexError, KeyError, TypeError, ValueError):
        return False
    return True


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
    canonical_policy_manifest: dict[str, str] | None = None,
) -> tuple[list[EditorialUnit], dict[str, Any]]:
    units: list[EditorialUnit] = []
    path_manifests: list[dict[str, Any]] = []
    protected_term_list = list(
        dict.fromkeys(
            str(item).strip() for item in protected_terms if str(item).strip()
        )
    )
    document_kind, reader_paths, default_paths, immutable_paths = (
        _editorial_path_contract(report)
    )
    selected_paths = (
        default_paths
        if prose_paths is None
        else _explicit_narrative_paths(report, prose_paths)
    )
    selected_set = set(selected_paths)
    immutable_set = set(immutable_paths)
    reader_set = set(reader_paths)
    overlap_paths = sorted(selected_set & immutable_set)
    missing_paths = sorted(reader_set - selected_set - immutable_set)
    unexpected_paths = sorted((selected_set | immutable_set) - reader_set)
    coverage_complete = not (overlap_paths or missing_paths or unexpected_paths)
    passthrough_receipts = [
        {
            "path": path,
            "source_sha256": text_sha256(str(_pointer_get(report, path))),
            "source_utf8_bytes": len(str(_pointer_get(report, path)).encode("utf-8")),
        }
        for path in immutable_paths
    ]
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
            code_owned_prefix, editable_text, code_owned_suffix = (
                _split_code_owned_edge_whitespace(item.text)
            )
            editable_start_in_context = item.core_start_in_context + len(
                code_owned_prefix
            )
            editable_end_in_context = item.core_end_in_context - len(code_owned_suffix)
            if (
                item.context_text[editable_start_in_context:editable_end_in_context]
                != editable_text
            ):
                raise RuntimeError("Editorial core is not embedded in its context")
            units.append(
                EditorialUnit(
                    unit_id=item.unit_id,
                    path=path,
                    index=item.index,
                    start_char=item.start_char,
                    end_char=item.end_char,
                    source_text=item.text,
                    editable_text=editable_text,
                    code_owned_prefix=code_owned_prefix,
                    code_owned_suffix=code_owned_suffix,
                    partition_sha256=item.sha256,
                    context_text=item.context_text,
                    core_start_in_context=editable_start_in_context,
                    core_end_in_context=editable_end_in_context,
                    source_sha256=text_sha256(editable_text),
                    claims=_claims(editable_text),
                    protected_terms=_protected_terms(
                        editable_text,
                        protected_term_list,
                    ),
                )
            )
    canonical_policy = copy.deepcopy(
        canonical_policy_manifest
        if isinstance(canonical_policy_manifest, dict)
        else LIVE_RUSSIAN_POLICY_MANIFEST.as_dict()
    )
    manifest_core = {
        "version": REPORT_EDITOR_HARNESS_VERSION,
        "policy_version": REPORT_EDITOR_POLICY_VERSION,
        "policy_sha256": editorial_policy_sha256(),
        "canonical_policy": canonical_policy,
        "protected_terms_sha256": _stable_sha256(protected_term_list),
        "source_report_sha256": _stable_sha256(report),
        "document_kind": document_kind,
        "unit_count": len(units),
        "unit_ids": [item.unit_id for item in units],
        "path_selection": (
            "reader_report_default" if prose_paths is None else "explicit_json_pointer"
        ),
        "reader_string_paths": reader_paths,
        "prose_paths": selected_paths,
        "immutable_passthrough": passthrough_receipts,
        "missing_paths": missing_paths,
        "unexpected_paths": unexpected_paths,
        "overlap_paths": overlap_paths,
        "path_manifests": path_manifests,
        "boundary_contract": {
            "version": REPORT_EDITOR_BOUNDARY_VERSION,
            "edge_whitespace_owner": "code",
            "model_returns_edge_whitespace": False,
            "units": [_boundary_manifest_record(unit) for unit in units],
        },
        "coverage_complete": coverage_complete,
    }
    return units, {**manifest_core, "manifest_sha256": _stable_sha256(manifest_core)}


def validate_editorial_cache(
    source: dict[str, Any],
    result: dict[str, Any],
    audit: dict[str, Any],
    *,
    prose_paths: Iterable[str] | None = None,
    protected_terms: Iterable[str] = (),
    canonical_policy_manifest: dict[str, str] | None = None,
) -> bool:
    """Accept a cached edit only when hashes and exact path coverage revalidate."""

    if (
        not isinstance(source, dict)
        or not isinstance(result, dict)
        or not isinstance(audit, dict)
    ):
        return False
    expected_canonical_policy = copy.deepcopy(
        canonical_policy_manifest
        if isinstance(canonical_policy_manifest, dict)
        else LIVE_RUSSIAN_POLICY_MANIFEST.as_dict()
    )
    try:
        _units, expected_manifest = build_editorial_units(
            source,
            prose_paths=prose_paths,
            protected_terms=protected_terms,
            canonical_policy_manifest=expected_canonical_policy,
        )
    except (TypeError, ValueError):
        return False
    if expected_manifest.get("coverage_complete") is not True:
        return False
    if audit.get("coverage_complete") is not True:
        return False
    if audit.get("path_coverage_complete") is not True:
        return False
    if audit.get("boundary_integrity_complete") is not True:
        return False
    if audit.get("quality_complete") is not True:
        return False
    if audit.get("fallback_units") != []:
        return False
    if audit.get("version") != REPORT_EDITOR_HARNESS_VERSION:
        return False
    if audit.get("policy_version") != REPORT_EDITOR_POLICY_VERSION:
        return False
    if audit.get("policy_sha256") != editorial_policy_sha256():
        return False
    if audit.get("canonical_policy") != expected_canonical_policy:
        return False
    if audit.get("source_report_sha256") != _stable_sha256(source):
        return False
    if audit.get("result_report_sha256") != _stable_sha256(result):
        return False
    if audit.get("source_manifest") != expected_manifest:
        return False
    if seal_editorial_audit(audit).get("audit_sha256") != audit.get("audit_sha256"):
        return False
    if audit.get("unit_count") != expected_manifest.get("unit_count"):
        return False
    if audit.get("processed_unit_count") != expected_manifest.get("unit_count"):
        return False
    critic_verdicts = audit.get("critic_verdicts")
    if not isinstance(critic_verdicts, list):
        return False
    if [row.get("unit_id") for row in critic_verdicts if isinstance(row, dict)] != (
        expected_manifest.get("unit_ids") or []
    ):
        return False
    if any(
        not isinstance(row, dict)
        or row.get("verdict") not in _ACCEPTED_EDITORIAL_VERDICTS
        for row in critic_verdicts
    ):
        return False
    if not _boundary_assembly_is_valid(
        _units,
        result,
        audit.get("boundary_receipts"),
    ):
        return False
    semantic_fallback = audit.get("semantic_fallback")
    if semantic_fallback is not None and (
        not isinstance(semantic_fallback, dict)
        or semantic_fallback.get("used") is not False
    ):
        return False
    copy_lint = audit.get("reader_copy_lint")
    if (
        not isinstance(copy_lint, dict)
        or copy_lint.get("policy_version") != expected_canonical_policy.get("version")
        or copy_lint.get("policy_sha256") != expected_canonical_policy.get("sha256")
        or copy_lint.get("blocking") is not False
        or copy_lint.get("issues") != []
        or copy_lint.get("omitted_issue_count") != 0
    ):
        return False
    if set(_all_string_leaf_paths(result)) != set(
        expected_manifest.get("reader_string_paths") or []
    ):
        return False

    editable_paths = set(expected_manifest.get("prose_paths") or [])
    actual_changed_paths: set[str] = set()
    try:
        for path in expected_manifest.get("reader_string_paths") or []:
            source_value = _pointer_get(source, path)
            result_value = _pointer_get(result, path)
            if source_value != result_value:
                actual_changed_paths.add(path)
        for receipt in expected_manifest.get("immutable_passthrough") or []:
            path = str(receipt["path"])
            source_value = _pointer_get(source, path)
            result_value = _pointer_get(result, path)
            if source_value != result_value:
                return False
            if receipt.get("source_sha256") != text_sha256(str(source_value)):
                return False
            if receipt.get("source_utf8_bytes") != len(
                str(source_value).encode("utf-8")
            ):
                return False
    except (IndexError, KeyError, TypeError, ValueError):
        return False
    if not actual_changed_paths.issubset(editable_paths):
        return False
    if set(audit.get("changed_paths") or []) != actual_changed_paths:
        return False
    return True


def _trusted_archived_editorial_contract(
    audit: dict[str, Any],
    source_manifest: dict[str, Any],
) -> EditorialContractManifest | None:
    """Resolve an old proof only from the explicit code-owned archive."""

    canonical_policy = audit.get("canonical_policy")
    if canonical_policy != source_manifest.get("canonical_policy"):
        return None
    if trusted_live_russian_policy_manifest(canonical_policy) is None:
        return None
    boundary_contract = source_manifest.get("boundary_contract")
    if not isinstance(boundary_contract, dict):
        return None
    path_manifests = source_manifest.get("path_manifests")
    if not isinstance(path_manifests, list) or not path_manifests:
        return None
    partition_versions = {
        row.get("manifest", {}).get("version")
        for row in path_manifests
        if isinstance(row, dict) and isinstance(row.get("manifest"), dict)
    }
    if len(partition_versions) != 1:
        return None
    actual = {
        "harness_version": audit.get("version"),
        "policy_version": audit.get("policy_version"),
        "policy_sha256": audit.get("policy_sha256"),
        "boundary_version": boundary_contract.get("version"),
        "partition_version": next(iter(partition_versions)),
        "canonical_policy": canonical_policy,
    }
    for candidate in TRUSTED_EDITORIAL_CONTRACT_MANIFESTS:
        descriptor = candidate.as_dict()
        descriptor.pop("accepted_verdicts", None)
        if actual == descriptor:
            return candidate
    return None


def _archived_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _archived_text_receipt_matches(
    text: str,
    receipt: dict[str, Any],
    *,
    chars_key: str,
    bytes_key: str,
    sha_key: str,
) -> bool:
    return bool(
        receipt.get(chars_key) == len(text)
        and receipt.get(bytes_key) == len(text.encode("utf-8"))
        and receipt.get(sha_key) == text_sha256(text)
    )


def validate_archived_editorial_cache(
    source: dict[str, Any],
    result: dict[str, Any],
    audit: dict[str, Any],
    *,
    prose_paths: Iterable[str] | None = None,
    protected_terms: Iterable[str] = (),
    canonical_policy_manifest: dict[str, str] | None = None,
) -> bool:
    """Verify a retained proof without re-running the current editor splitter.

    Staging uses :func:`validate_editorial_cache` and therefore enforces the
    current policy and algorithm.  A public read instead verifies the exact
    historical source/result partitions, self-seals and a code-owned archived
    contract.  This preserves old immutable publications across a deliberate
    editor upgrade while unknown proof formats still fail closed.
    """

    if not all(isinstance(value, dict) for value in (source, result, audit)):
        return False
    source_manifest = audit.get("source_manifest")
    if not isinstance(source_manifest, dict):
        return False
    contract = _trusted_archived_editorial_contract(audit, source_manifest)
    if contract is None:
        return False
    canonical_policy = contract.as_dict()["canonical_policy"]
    if (
        canonical_policy_manifest is not None
        and canonical_policy_manifest != canonical_policy
    ):
        return False

    audit_core = copy.deepcopy(audit)
    audit_digest = audit_core.pop("audit_sha256", None)
    manifest_core = copy.deepcopy(source_manifest)
    manifest_digest = manifest_core.pop("manifest_sha256", None)
    unit_count = _archived_nonnegative_int(audit.get("unit_count"))
    if (
        audit_digest != _stable_sha256(audit_core)
        or manifest_digest != _stable_sha256(manifest_core)
        or audit.get("source_report_sha256") != _stable_sha256(source)
        or audit.get("result_report_sha256") != _stable_sha256(result)
        or source_manifest.get("source_report_sha256") != _stable_sha256(source)
        or source_manifest.get("version") != contract.harness_version
        or source_manifest.get("policy_version") != contract.policy_version
        or source_manifest.get("policy_sha256") != contract.policy_sha256
        or source_manifest.get("canonical_policy") != canonical_policy
        or unit_count is None
        or source_manifest.get("unit_count") != unit_count
        or audit.get("processed_unit_count") != unit_count
        or audit.get("fallback_units") != []
        or audit.get("path_coverage_complete") is not True
        or audit.get("boundary_integrity_complete") is not True
        or audit.get("quality_complete") is not True
        or audit.get("coverage_complete") is not True
        or source_manifest.get("coverage_complete") is not True
        or source_manifest.get("missing_paths") != []
        or source_manifest.get("unexpected_paths") != []
        or source_manifest.get("overlap_paths") != []
    ):
        return False

    protected_term_list = [
        str(item).strip() for item in protected_terms if str(item).strip()
    ]
    protected_term_list = list(dict.fromkeys(protected_term_list))
    if source_manifest.get("protected_terms_sha256") != _stable_sha256(
        protected_term_list
    ):
        return False
    reader_paths = source_manifest.get("reader_string_paths")
    selected_paths = source_manifest.get("prose_paths")
    passthrough = source_manifest.get("immutable_passthrough")
    if (
        not isinstance(reader_paths, list)
        or any(not isinstance(path, str) or not path for path in reader_paths)
        or len(reader_paths) != len(set(reader_paths))
        or not isinstance(selected_paths, list)
        or any(not isinstance(path, str) or not path for path in selected_paths)
        or len(selected_paths) != len(set(selected_paths))
        or not isinstance(passthrough, list)
    ):
        return False
    if set(_all_string_leaf_paths(source)) != set(reader_paths):
        return False
    if set(_all_string_leaf_paths(result)) != set(reader_paths):
        return False
    if prose_paths is None:
        if source_manifest.get("path_selection") != "reader_report_default":
            return False
    else:
        supplied_paths = list(dict.fromkeys(prose_paths))
        if (
            source_manifest.get("path_selection") != "explicit_json_pointer"
            or supplied_paths != selected_paths
        ):
            return False

    passthrough_paths: list[str] = []
    try:
        for receipt in passthrough:
            if not isinstance(receipt, dict):
                return False
            path = receipt.get("path")
            if not isinstance(path, str) or not path:
                return False
            source_value = _pointer_get(source, path)
            result_value = _pointer_get(result, path)
            if (
                not isinstance(source_value, str)
                or result_value != source_value
                or receipt.get("source_sha256") != text_sha256(source_value)
                or receipt.get("source_utf8_bytes") != len(source_value.encode("utf-8"))
            ):
                return False
            passthrough_paths.append(path)
    except (IndexError, KeyError, TypeError, ValueError):
        return False
    if (
        len(passthrough_paths) != len(set(passthrough_paths))
        or set(selected_paths) & set(passthrough_paths)
        or set(selected_paths) | set(passthrough_paths) != set(reader_paths)
    ):
        return False

    path_manifests = source_manifest.get("path_manifests")
    boundary_contract = source_manifest.get("boundary_contract")
    boundary_rows = (
        boundary_contract.get("units") if isinstance(boundary_contract, dict) else None
    )
    if (
        not isinstance(path_manifests, list)
        or len(path_manifests) != len(selected_paths)
        or not isinstance(boundary_rows, list)
        or boundary_contract.get("edge_whitespace_owner") != "code"
        or boundary_contract.get("model_returns_edge_whitespace") is not False
    ):
        return False

    source_units: list[dict[str, Any]] = []
    try:
        for path, row in zip(selected_paths, path_manifests, strict=True):
            if not isinstance(row, dict) or row.get("path") != path:
                return False
            manifest = row.get("manifest")
            source_text = _pointer_get(source, path)
            if not isinstance(manifest, dict) or not isinstance(source_text, str):
                return False
            manifest_units = manifest.get("units")
            manifest_unit_count = _archived_nonnegative_int(manifest.get("unit_count"))
            if (
                manifest.get("version") != contract.partition_version
                or manifest.get("source_sha256") != text_sha256(source_text)
                or manifest.get("source_chars") != len(source_text)
                or manifest.get("source_utf8_bytes") != len(source_text.encode("utf-8"))
                or manifest_unit_count is None
                or not isinstance(manifest_units, list)
                or len(manifest_units) != manifest_unit_count
            ):
                return False
            cursor = 0
            for index, unit in enumerate(manifest_units):
                if not isinstance(unit, dict):
                    return False
                start = _archived_nonnegative_int(unit.get("start_char"))
                end = _archived_nonnegative_int(unit.get("end_char"))
                context_start = _archived_nonnegative_int(
                    unit.get("context_start_char")
                )
                context_end = _archived_nonnegative_int(unit.get("context_end_char"))
                if (
                    start != cursor
                    or end is None
                    or context_start is None
                    or context_end is None
                    or not (0 <= context_start <= start <= end <= context_end)
                    or context_end > len(source_text)
                    or unit.get("index") != index
                ):
                    return False
                partition = source_text[start:end]
                context = source_text[context_start:context_end]
                if (
                    end - start != len(partition)
                    or unit.get("utf8_bytes") != len(partition.encode("utf-8"))
                    or unit.get("sha256") != text_sha256(partition)
                ):
                    return False
                if (
                    unit.get("context_start_char") != context_start
                    or unit.get("context_end_char") != context_end
                    or unit.get("context_utf8_bytes") != len(context.encode("utf-8"))
                    or unit.get("context_sha256") != text_sha256(context)
                    or unit.get("core_start_in_context") != start - context_start
                    or unit.get("core_end_in_context") != end - context_start
                ):
                    return False
                source_units.append(
                    {
                        "path": path,
                        "index": index,
                        "unit_id": unit.get("unit_id"),
                        "partition": partition,
                        "partition_sha256": unit.get("sha256"),
                    }
                )
                cursor = end
            if cursor != len(source_text):
                return False
    except (IndexError, KeyError, TypeError, ValueError):
        return False

    unit_ids = [row.get("unit_id") for row in source_units]
    if (
        len(source_units) != unit_count
        or any(not isinstance(unit_id, str) or not unit_id for unit_id in unit_ids)
        or len(unit_ids) != len(set(unit_ids))
        or source_manifest.get("unit_ids") != unit_ids
        or len(boundary_rows) != unit_count
    ):
        return False

    receipts = audit.get("boundary_receipts")
    critic_rows = audit.get("critic_verdicts")
    if (
        not isinstance(receipts, list)
        or len(receipts) != unit_count
        or not isinstance(critic_rows, list)
        or len(critic_rows) != unit_count
    ):
        return False
    if [row.get("unit_id") for row in critic_rows if isinstance(row, dict)] != unit_ids:
        return False
    if any(
        not isinstance(row, dict)
        or row.get("verdict") not in contract.accepted_verdicts
        for row in critic_rows
    ):
        return False

    result_cursors: dict[str, int] = defaultdict(int)
    touched_paths: set[str] = set()
    try:
        for unit, boundary, receipt in zip(
            source_units,
            boundary_rows,
            receipts,
            strict=True,
        ):
            if not isinstance(boundary, dict) or not isinstance(receipt, dict):
                return False
            path = unit["path"]
            index = unit["index"]
            unit_id = unit["unit_id"]
            partition = unit["partition"]
            if any(
                row.get("unit_id") != unit_id
                or row.get("path") != path
                or row.get("index") != index
                or row.get("partition_sha256") != unit["partition_sha256"]
                for row in (boundary, receipt)
            ):
                return False
            if receipt.get("version") != contract.boundary_version:
                return False
            prefix_chars = _archived_nonnegative_int(
                boundary.get("code_owned_prefix_chars")
            )
            suffix_chars = _archived_nonnegative_int(
                boundary.get("code_owned_suffix_chars")
            )
            assembled_chars = _archived_nonnegative_int(receipt.get("assembled_chars"))
            editable_chars = _archived_nonnegative_int(
                receipt.get("editable_result_chars")
            )
            if (
                prefix_chars is None
                or suffix_chars is None
                or assembled_chars is None
                or editable_chars is None
                or prefix_chars + suffix_chars > len(partition)
                or assembled_chars != prefix_chars + editable_chars + suffix_chars
            ):
                return False
            suffix_start = (
                len(partition) - suffix_chars if suffix_chars else len(partition)
            )
            prefix = partition[:prefix_chars]
            editable_source = partition[prefix_chars:suffix_start]
            suffix = partition[suffix_start:]
            if (
                (prefix and not prefix.isspace())
                or (suffix and not suffix.isspace())
                or boundary.get("code_owned_prefix_utf8_bytes")
                != len(prefix.encode("utf-8"))
                or boundary.get("code_owned_prefix_sha256") != text_sha256(prefix)
                or boundary.get("code_owned_suffix_utf8_bytes")
                != len(suffix.encode("utf-8"))
                or boundary.get("code_owned_suffix_sha256") != text_sha256(suffix)
                or boundary.get("editable_source_sha256")
                != text_sha256(editable_source)
                or receipt.get("code_owned_prefix_chars") != prefix_chars
                or receipt.get("code_owned_prefix_sha256") != text_sha256(prefix)
                or receipt.get("code_owned_suffix_chars") != suffix_chars
                or receipt.get("code_owned_suffix_sha256") != text_sha256(suffix)
            ):
                return False
            result_text = _pointer_get(result, path)
            if not isinstance(result_text, str):
                return False
            start = result_cursors[path]
            assembled = result_text[start : start + assembled_chars]
            if not _archived_text_receipt_matches(
                assembled,
                receipt,
                chars_key="assembled_chars",
                bytes_key="assembled_utf8_bytes",
                sha_key="assembled_sha256",
            ):
                return False
            editable_result = assembled[
                prefix_chars : len(assembled) - suffix_chars
                if suffix_chars
                else len(assembled)
            ]
            if (
                not assembled.startswith(prefix)
                or not assembled.endswith(suffix)
                or editable_result != editable_result.strip()
                or not _archived_text_receipt_matches(
                    editable_result,
                    receipt,
                    chars_key="editable_result_chars",
                    bytes_key="editable_result_utf8_bytes",
                    sha_key="editable_result_sha256",
                )
            ):
                return False
            result_cursors[path] += assembled_chars
            touched_paths.add(path)
    except (IndexError, KeyError, TypeError, ValueError):
        return False
    try:
        if any(
            result_cursors[path] != len(str(_pointer_get(result, path)))
            for path in touched_paths
        ):
            return False
        actual_changed_paths = {
            path
            for path in reader_paths
            if _pointer_get(source, path) != _pointer_get(result, path)
        }
    except (IndexError, KeyError, TypeError, ValueError):
        return False
    if (
        not actual_changed_paths.issubset(set(selected_paths))
        or set(audit.get("changed_paths") or []) != actual_changed_paths
    ):
        return False

    semantic_fallback = audit.get("semantic_fallback")
    if semantic_fallback is not None and (
        not isinstance(semantic_fallback, dict)
        or semantic_fallback.get("used") is not False
    ):
        return False
    copy_lint = audit.get("reader_copy_lint")
    if (
        not isinstance(copy_lint, dict)
        or copy_lint.get("policy_version") != canonical_policy.get("version")
        or copy_lint.get("policy_sha256") != canonical_policy.get("sha256")
        or copy_lint.get("blocking") is not False
        or copy_lint.get("issues") != []
        or copy_lint.get("omitted_issue_count") != 0
    ):
        return False
    return True


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


def _relational_fact_bindings(
    text: str,
    protected_terms: Iterable[str],
    *,
    fact_pattern: re.Pattern[str],
    normalize_fact: Callable[[str], str],
) -> Counter[tuple[str, str]]:
    """Bind repeated facts to the nearest named entity in the text.

    Global multisets catch deleted values but not `OpenAI: 4; Gemini: 2`
    becoming `OpenAI: 2; Gemini: 4`.  These deterministic bindings reject that
    swap while still allowing the editor to reorder clauses or sentences.
    """

    folded = text.casefold()
    url_spans = [match.span() for match in _URL_RE.finditer(text)]
    actor_candidates: list[tuple[int, int, str]] = []
    for raw_term in protected_terms:
        term = str(raw_term or "").strip()
        if not term:
            continue
        for match in re.finditer(re.escape(term.casefold()), folded):
            if any(
                url_start <= match.start() and match.end() <= url_end
                for url_start, url_end in url_spans
            ):
                continue
            actor_candidates.append((match.start(), match.end(), term.casefold()))
    # Prefer the longest protected name when aliases overlap at one position.
    actors: list[tuple[int, int, str]] = []
    for candidate in sorted(
        actor_candidates,
        key=lambda value: (value[0], -(value[1] - value[0]), value[2]),
    ):
        start, end, _name = candidate
        if any(
            start < kept_end and kept_start < end for kept_start, kept_end, _ in actors
        ):
            continue
        actors.append(candidate)

    facts = [
        (match.start(), match.end(), normalize_fact(match.group(0)))
        for match in fact_pattern.finditer(text)
    ]
    if len({actor[2] for actor in actors}) < 2 or len(facts) < 2:
        return Counter()

    sentence_source = list(text)
    for raw_start, raw_end in url_spans:
        url_end = raw_end
        while url_end > raw_start and text[url_end - 1] in ".,;:!?":
            url_end -= 1
        for index in range(raw_start, url_end):
            if sentence_source[index] in ".!?":
                sentence_source[index] = "_"
    sentence_ranges = [
        (match.start(), match.end())
        for match in _SENTENCE_RE.finditer("".join(sentence_source))
    ]
    bindings: list[tuple[str, str]] = []
    for fact_start, fact_end, fact_value in facts:
        sentence_start, sentence_end = next(
            (
                bounds
                for bounds in sentence_ranges
                if bounds[0] <= fact_start < bounds[1]
            ),
            (0, len(text)),
        )
        local_actors = [
            actor
            for actor in actors
            if sentence_start <= actor[0] and actor[1] <= sentence_end
        ]
        if not local_actors:
            bindings.append((fact_value, ""))
            continue
        distances = [
            (
                max(actor_start - fact_end, fact_start - actor_end, 0),
                actor_name,
            )
            for actor_start, actor_end, actor_name in local_actors
        ]
        nearest_distance = min(distance for distance, _name in distances)
        nearest_names = sorted(
            {name for distance, name in distances if distance == nearest_distance}
        )
        bindings.append((fact_value, "|".join(nearest_names)))
    return Counter(bindings)


def _actor_value_bindings(
    text: str,
    protected_terms: Iterable[str],
) -> Counter[tuple[str, str]]:
    return _relational_fact_bindings(
        text,
        protected_terms,
        fact_pattern=_NUMBER_RE,
        normalize_fact=lambda value: re.sub(
            r"\s+",
            " ",
            re.sub(r"(?<=\d),(?=\d)", ".", value.strip().casefold()),
        ),
    )


def _actor_url_bindings(
    text: str,
    protected_terms: Iterable[str],
) -> Counter[tuple[str, str]]:
    return _relational_fact_bindings(
        text,
        protected_terms,
        fact_pattern=_URL_RE,
        normalize_fact=lambda value: value.casefold(),
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
    if not isinstance(edited, str) or (
        unit.editable_text.strip() and not edited.strip()
    ):
        errors.append("edited_text_missing")
        edited = ""
    if edited != edited.strip():
        errors.append("code_owned_boundary_changed")
    if not unit.editable_text and edited:
        errors.append("code_owned_only_unit_changed")
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
    if Counter(_URL_RE.findall(unit.editable_text)) != Counter(_URL_RE.findall(edited)):
        errors.append("url_set_changed")
    if _normalized_numbers(unit.editable_text) != _normalized_numbers(edited):
        errors.append("number_or_unit_set_changed")
    if _literal_counts(unit.editable_text, unit.protected_terms) != _literal_counts(
        edited, unit.protected_terms
    ):
        errors.append("protected_name_set_changed")
    if _actor_value_bindings(
        unit.editable_text,
        unit.protected_terms,
    ) != _actor_value_bindings(edited, unit.protected_terms):
        errors.append("actor_value_binding_changed")
    if _actor_url_bindings(
        unit.editable_text,
        unit.protected_terms,
    ) != _actor_url_bindings(edited, unit.protected_terms):
        errors.append("actor_url_binding_changed")
    if _INTERNAL_META_RE.search(edited) and not _INTERNAL_META_RE.search(
        unit.editable_text
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
    for issue in lint_russian_copy(edited).issues:
        errors.append(f"live_russian_{issue.code}")
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
                "meaning_preserved",
                "actor_preserved",
                "scope_preserved",
                "numbers_preserved",
            )
        ):
            errors.append("critic_found_semantic_drift")
        if not isinstance(item, dict) or not all(
            item.get(key) is True
            for key in (
                "actor_or_mechanism_explicit",
                "number_carrier_explicit",
                "active_voice",
                "no_slogan_or_meta",
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
    if manifest.get("coverage_complete") is not True:
        audit = EditorialAudit(
            version=REPORT_EDITOR_HARNESS_VERSION,
            policy_version=REPORT_EDITOR_POLICY_VERSION,
            policy_sha256=editorial_policy_sha256(),
            source_report_sha256=manifest["source_report_sha256"],
            result_report_sha256=_stable_sha256(source),
            unit_count=len(units),
            processed_unit_count=0,
            changed_paths=(),
            fallback_units=(
                {
                    "unit_id": "*",
                    "verdict": "fallback",
                    "reason": "reader_string_coverage_incomplete",
                },
            ),
            critic_verdicts=(),
            boundary_receipts=(),
            path_coverage_complete=False,
            boundary_integrity_complete=False,
            quality_complete=False,
            coverage_complete=False,
        ).as_dict()
        audit["source_manifest"] = manifest
        return source, seal_editorial_audit(audit)
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
                    "source_text": unit.editable_text,
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
                                "source_text": unit.editable_text,
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
            unit.editable_text,
            {
                "unit_id": unit.unit_id,
                "verdict": "fallback",
                "reason": ";".join(errors) or "editorial_validation_failed",
            },
        )

    outcomes = await asyncio.gather(*(process(unit) for unit in units))
    by_id = {unit_id: (text, audit) for unit_id, text, audit in outcomes}
    rebuilt: dict[str, list[tuple[int, EditorialUnit, str]]] = defaultdict(list)
    critic_rows: list[dict[str, str]] = []
    fallback_rows: list[dict[str, str]] = []
    for unit in units:
        edited, row = by_id[unit.unit_id]
        rebuilt[unit.path].append((unit.index, unit, edited))
        critic_rows.append(row)
        if row.get("verdict") == "fallback":
            fallback_rows.append(row)

    result = copy.deepcopy(source)
    changed_paths: list[str] = []
    boundary_receipts: list[dict[str, Any]] = []
    boundary_integrity_complete = True
    for path, parts in rebuilt.items():
        ordered_parts = sorted(parts, key=lambda row: row[0])
        if [index for index, _unit, _text in ordered_parts] != list(
            range(len(ordered_parts))
        ):
            boundary_integrity_complete = False
            continue
        fragments: list[str] = []
        try:
            for _index, unit, edited_text in ordered_parts:
                receipt = _boundary_receipt(unit, edited_text)
                boundary_receipts.append(receipt)
                fragments.append(
                    unit.code_owned_prefix + edited_text + unit.code_owned_suffix
                )
        except ValueError:
            boundary_integrity_complete = False
            continue
        value = "".join(fragments)
        if value != _pointer_get(source, path):
            changed_paths.append(path)
        _pointer_set(result, path, value)
    processed_ids = [row[0] for row in outcomes]
    path_coverage_complete = (
        len(processed_ids) == len(units)
        and len(set(processed_ids)) == len(units)
        and set(processed_ids) == {unit.unit_id for unit in units}
    )
    boundary_integrity_complete = (
        path_coverage_complete
        and boundary_integrity_complete
        and _boundary_assembly_is_valid(units, result, boundary_receipts)
    )
    if not path_coverage_complete or not boundary_integrity_complete:
        result = source
        changed_paths = []
        fallback_rows.append(
            {
                "unit_id": "*",
                "verdict": "fallback",
                "reason": (
                    "boundary_integrity_incomplete"
                    if path_coverage_complete
                    else "coverage_incomplete"
                ),
            }
        )
    quality_complete = (
        path_coverage_complete
        and boundary_integrity_complete
        and not fallback_rows
        and len(critic_rows) == len(units)
        and all(
            row.get("verdict") in _ACCEPTED_EDITORIAL_VERDICTS for row in critic_rows
        )
    )
    audit = EditorialAudit(
        version=REPORT_EDITOR_HARNESS_VERSION,
        policy_version=REPORT_EDITOR_POLICY_VERSION,
        policy_sha256=editorial_policy_sha256(),
        source_report_sha256=manifest["source_report_sha256"],
        result_report_sha256=_stable_sha256(result),
        unit_count=len(units),
        processed_unit_count=len(processed_ids),
        changed_paths=tuple(sorted(changed_paths)),
        fallback_units=tuple(copy.deepcopy(fallback_rows)),
        critic_verdicts=tuple(copy.deepcopy(critic_rows)),
        boundary_receipts=tuple(copy.deepcopy(boundary_receipts)),
        path_coverage_complete=path_coverage_complete,
        boundary_integrity_complete=boundary_integrity_complete,
        quality_complete=quality_complete,
        coverage_complete=(path_coverage_complete and boundary_integrity_complete),
    ).as_dict()
    audit["source_manifest"] = manifest
    audit["reader_copy_lint"] = lint_reader_copy_tree(result).as_dict()
    return result, seal_editorial_audit(audit)
