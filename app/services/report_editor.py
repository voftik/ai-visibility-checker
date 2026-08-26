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


REPORT_EDITOR_POLICY_VERSION = "aiv-ru-editorial-policy-v3"
REPORT_EDITOR_HARNESS_VERSION = "aiv-report-editor-lossless-v3"
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


def seal_editorial_audit(audit: dict[str, Any]) -> dict[str, Any]:
    """Bind every persisted audit field, including the coverage manifest."""

    value = copy.deepcopy(audit)
    value.pop("audit_sha256", None)
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
        paths.extend(
            f"/actions/{index}/{field}" for field in ("title", "why", "step")
        )
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
            paths.extend(
                (f"/actions/{index}/priority", f"/actions/{index}/evidence")
            )
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
            paths.extend(
                (f"/findings/{index}/severity", f"/findings/{index}/evidence")
            )
    return _existing_nonempty_string_paths(review, paths)


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
    else:
        raise ValueError("unsupported editorial document shape")

    # Empty schema strings have no model unit, but are still covered and bound.
    immutable_set = set(immutable)
    for path in expected:
        if path not in editable and _pointer_get(document, path) == "":
            immutable_set.add(path)
    return kind, expected, editable, [path for path in expected if path in immutable_set]


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
    protected_term_list = list(
        dict.fromkeys(str(item).strip() for item in protected_terms if str(item).strip())
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
                    protected_terms=_protected_terms(item.text, protected_term_list),
                )
            )
    manifest_core = {
        "version": REPORT_EDITOR_HARNESS_VERSION,
        "policy_version": REPORT_EDITOR_POLICY_VERSION,
        "policy_sha256": editorial_policy_sha256(),
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
) -> bool:
    """Accept a cached edit only when hashes and exact path coverage revalidate."""

    if not isinstance(source, dict) or not isinstance(result, dict) or not isinstance(
        audit, dict
    ):
        return False
    try:
        _units, expected_manifest = build_editorial_units(
            source,
            prose_paths=prose_paths,
            protected_terms=protected_terms,
        )
    except (TypeError, ValueError):
        return False
    if expected_manifest.get("coverage_complete") is not True:
        return False
    if audit.get("coverage_complete") is not True:
        return False
    if audit.get("version") != REPORT_EDITOR_HARNESS_VERSION:
        return False
    if audit.get("policy_version") != REPORT_EDITOR_POLICY_VERSION:
        return False
    if audit.get("policy_sha256") != editorial_policy_sha256():
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
        policy_sha256=editorial_policy_sha256(),
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
    return result, seal_editorial_audit(audit)
