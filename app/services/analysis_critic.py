from __future__ import annotations

import hashlib
import json
from typing import Any

from app.config import settings
from app.services.openrouter import (
    OpenRouterError,
    OpenRouterOutputLimitError,
    OpenRouterResponseContractError,
    chat,
)

CRITIC_VERSION = "aiv-analysis-critic-v19"
CRITIC_TRANSPORT_CONTRACT_VERSION = "aiv-analysis-critic-transport-v1"
MAX_CRITIC_ITERATIONS = 2
MAX_CRITIC_RECOVERY_FINAL_REVIEWS = 1
MAX_CRITIC_REPAIR_ATTEMPTS = 1
CRITIC_MODEL = settings.OPENROUTER_CRITIC_MODEL
CRITIC_REASONING_EFFORT = "medium"
CRITIC_MAX_TOKENS = 20_000
CRITIC_REPAIR_REASONING_EFFORT = "low"
CRITIC_REPAIR_MAX_TOKENS = 8_000
CRITIC_REPAIR_MAX_AFFECTED_ANSWERS = 12
CRITIC_REPAIR_RAW_CHAR_LIMIT = 6_000
CRITIC_PARTIAL_RESPONSE_CHAR_LIMIT = 16_000
CRITIC_PRIMARY_RAW_CHAR_BUDGET = 120_000
CRITIC_PRIMARY_MAX_RAW_ANSWERS = 24

CRITIC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["pass", "revise", "block"],
        },
        "summary": {"type": "string"},
        "anomalies": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "code": {
                        "type": "string",
                        "enum": [
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
                        ],
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["critical", "important", "observation"],
                    },
                    "finding": {"type": "string"},
                    "answer_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                    "entities": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "code",
                    "severity",
                    "finding",
                    "answer_ids",
                    "entities",
                ],
            },
        },
        "policy_adjustments": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "exclude_portfolio_entity",
                            "require_target_attribution",
                            "require_alias_attribution",
                            "require_literal_attribution_evidence",
                            "require_literal_brand_knowledge_evidence",
                        ],
                    },
                    "entity_name": {"type": "string"},
                    "alias": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "null"},
                        ]
                    },
                    "reason": {"type": "string"},
                    "answer_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                },
                "required": [
                    "action",
                    "entity_name",
                    "alias",
                    "reason",
                    "answer_ids",
                ],
            },
        },
        "annotation_guidance": {"type": "string"},
        "acceptance_checks": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "verdict",
        "summary",
        "anomalies",
        "policy_adjustments",
        "annotation_guidance",
        "acceptance_checks",
    ],
}

CRITIC_SYSTEM = """
Ты независимый критик-методолог отчёта AI visibility. Проверяй не красоту
вывода, а происхождение каждой метрики из исходных ответов. Тебе переданы
профиль исследуемого сайта, каталог сущностей, сценарии, полный индекс
raw-ответов с SHA, разметка и уже рассчитанные срезы. Длинный raw-текст
передаётся в первую очередь для строк из deterministic_warnings, затем для
детерминированной стратифицированной выборки положительных, отрицательных и
ошибочных исходов, в пределах общего char budget. Остальные строки остаются в
полном manifest с annotation/provenance и raw_answer_included=false. Плановое
отсутствие обычной выборочной строки не требует block. Но если
raw_evidence_selection.omitted_warning_answer_ids или
missing_warning_answer_ids не пуст, материальному предупреждению не хватило
raw-контекста: выбери block.

Главные инварианты:
- metric_evidence_state=legacy_observational означает ограниченный
  исторический memory-срез: точная модель подтверждена, онлайн-вариант,
  ссылки и сигналы web/tool не наблюдались, но явное отключение веба не было
  сохранено. Такие строки разрешено проверять и считать в агрегатах только с
  data_state=limited; нельзя выдавать их за строго подтверждённый no-web
  режим, использовать как причинное доказательство эффекта веб-поиска или
  требовать их обнуления только из-за отсутствия нового transport bundle;
- context_eligible=false не отменяет metric_eligible=true: содержимое такой
  legacy-observational строки может проверять только этот critic, а финальный
  автор получает агрегаты и метаданные без raw-текста;
- сначала учитывай metric_contract: portfolio_visibility и parent_discovery
  используют только scenario_role=unbranded_discovery. Для
  scenario_role=brand_diagnostic поле entity_mentions не входит в эти
  числители; там проверяй brand_answer и brand_knowledge. Не блокируй
  портфельную метрику из-за вспомогательного entity_mentions брендового ответа,
  если он не влияет на рассчитанный срез;
- безбрендовое обнаружение материнского бренда и обнаружение его продуктов —
  разные события и не могут наследовать флаги друг друга;
- конкретное собственное имя продукта можно засчитать по буквальному
  упоминанию, только если принадлежность продукту подтверждена сайтом;
- attributed_to_target описывает только буквальную связь внутри самого
  raw-ответа. Для подтверждённой сайтом сущности с mention_policy=standalone
  это поле может корректно оставаться false, когда материнский бренд в ответе
  не назван: такое значение не отменяет portfolio_visibility. Не требуй
  attributed_to_target=true и не предлагай require_target_attribution как
  исправление ложного отрицания для такого собственного имени;
- общая услуга или тема — доставка, аналитика, консалтинг, страхование,
  креатив и подобные —
  считается продуктовым попаданием лишь когда буквальный фрагмент ответа явно
  относит её к целевому бренду;
- совместное появление слов, нахождение в одном списке, предложении или
  абзаце не доказывает принадлежность;
- утверждения самой отвечающей модели не могут создать принадлежность
  сущности целевому портфелю, если сайт её не подтверждает;
- сущность с category=other, target_relationship=unrelated или
  _profile_membership_confirmed=false не входит в целевой портфель, даже если
  исходный ответ или вспомогательная разметка заявляет обратное;
- ошибочный attributed_to_target у сущности, которая по metric_contract и
  entity_catalog всё равно исключена из числителя, сам по себе не искажает
  метрику. Отметь это как observation; не выбирай revise только ради очистки
  вспомогательного поля, которое не участвует в расчёте;
- unknown, пропуск ответа и неполная разметка не превращаются в ноль;
- одинаковые доли у нескольких систем сами по себе возможны, но требуют
  проверки числителей и конкретных answer_id. Если две смыслово разные серии
  совпадают, ищи утечку scope или общий флаг;
- evidence должен быть точным непрерывным фрагментом raw-ответа: все символы
  внутри выбранных границ сохраняются без нормализации, исправлений и склейки.
  Границы могут не включать окружающие Markdown-маркеры, но фрагмент обязан
  содержать полное имя или алиас сущности;
- для общей portfolio-сущности владельцем могут быть только алиасы из
  entity_attribution_aliases[canonical_name]. Общий upstream owner из
  attribution_owner_aliases не переносит услуги или свойства соседнего
  продукта. Упоминание другого бренда, девелопера или компании не создаёт
  атрибуцию;
- если хотя бы у одного включённого ответа raw_answer_truncated=true,
  полного контекста
  для проверки нет: выбери block, не pass и не revise.

Проверяй и ложные отрицания. Если рассчитанный portfolio равен нулю или резко
ниже буквального содержания raw-ответов, найди конкретные answer_id, где один
из attribution_owner_aliases явно связан с подтверждённой услугой. Если
строгий фильтр не принял такие строки лишь потому, что разметчик скопировал
слишком короткий evidence, предложи require_literal_attribution_evidence:
повторная разметка должна скопировать один непрерывный буквальный фрагмент,
содержащий разрешённого владельца, услугу и выраженную связь между ними. Это не
расширение scope и не разрешение считать простое соседство.
Если непрерывного точного фрагмента или однозначного структурного контекста
заголовок→дочерний пункт нет, attributed_to_target=false и исключение из
числителя корректны. Не требуй синтезировать доказательство и не выбирай
revise лишь потому, что услуга и владелец встречаются где-то в одном ответе.

Отдельно проверь branded-сценарии. Если raw-ответ действительно сообщает
конкретные факты о целевом бренде, а brand_answer ошибочно помечен generic,
none или not_applicable, предложи require_literal_brand_knowledge_evidence
для конкретных answer_id. В brand_diagnostic значение not_applicable всегда
нарушает контракт, даже если вопрос сравнивает бренд с конкурентами. Такая
правка разрешена только когда факты буквальны, относятся
к целевому бренду и согласуются с site_profile или entity_catalog; одного
повтора названия из вопроса недостаточно.

Если deterministic_warnings сообщает, что разметка поставила
attributed_to_target=true, но буквальная проверка evidence отклонила много
таких связок, обязательно разберись по raw-ответам: это либо выдуманный
evidence, либо ложное отрицание из-за слишком короткого или неточного
фрагмента. Не ставь pass, пока важное расхождение не объяснено и не устранено.

Ты можешь только УЖЕСТОЧАТЬ текущую исследовательскую политику:
исключить неподтверждённую portfolio-сущность, потребовать явную атрибуцию
для сущности или отдельного alias либо потребовать полный буквальный фрагмент
атрибуции при повторной разметке. Для branded-сценария можно потребовать
повторную буквальную проверку конкретных фактов о бренде. Не предлагай менять
raw-ответы, знаменатели, числители или готовые цифры вручную. Не расширяй
portfolio scope. Каждая правка должна ссылаться на конкретные answer_id; при
отсутствии достаточных данных выбери block.

require_target_attribution и require_alias_attribution — только сужающие
действия против ложноположительного засчёта. Они не могут превратить false в
true и не исправляют потерю подтверждённого standalone-продукта. Сначала
проверь, не вызвана ли такая потеря конфликтом одноимённых записей каталога;
после детерминированного разрешения canonical_name оцени метрику заново.

verdict=pass ставь только если критических и важных ошибок не осталось.
При pass верни policy_adjustments=[] и annotation_guidance="" — не пиши туда
«изменения не нужны» или общие рекомендации. Верни хотя бы одну конкретную
acceptance_check. Каждую deterministic_warning с severity=critical/important
явно закрой anomaly с тем же code и severity=observation, объяснив, почему
она не влияет на опубликованный расчёт; перенеси в неё все answer_ids и
entities исходного предупреждения. Сама фраза критика не является
доказательством: pass откроется только если кодовый детерминированный resolver
независимо пересоберёт затронутые векторы из raw и канонической разметки.
Предупреждение без такого resolver требует revise или block. Наблюдения без
влияния на корректность можно оставить только с severity=observation.
verdict=revise — когда перечисленные ограниченные правки позволяют безопасно
повторить разметку и расчёт. verdict=block — если исходные данные повреждены,
неполны или корректность нельзя восстановить разрешёнными правками.
Пиши кратко и предметно по-русски.
""".strip()

CRITIC_REPAIR_SYSTEM = """
Ты исправляешь только контракт решения независимого критика AI visibility.
Первичный критик уже изучил полный набор исходных ответов, но его JSON-решение
оказалось семантически неполным или неприменимым. Не проводи новое исследование
и не придумывай новые факты. Сверь исходное решение с repair_context и верни
самодостаточное исправленное решение в той же строгой схеме. repair_context
содержит digest полного неизменяемого audit payload, каталог, метрики, индекс
всех ответов и raw только для ответов, прямо затронутых предупреждениями или
исходным решением. Если для безопасной правки не хватает показанного evidence,
выбери block; не пытайся восстановить пропущенные факты по догадке.

Правила восстановления:
- сначала проверь по metric_contract и candidate_metrics, влияет ли каждая
  anomaly на опубликованный числитель, знаменатель или состояние missing;
- неверный вспомогательный флаг у entity, которая уже исключена из метрики
  каталогом, может остаться только observation и не требует revise;
- verdict=pass допустим, только если после такой проверки не осталось
  critical/important anomaly; при pass policy_adjustments=[],
  annotation_guidance="" и есть хотя бы одна конкретная acceptance_check.
  Каждая critical/important deterministic_warning должна быть явно закрыта
  содержательной anomaly с тем же code и severity=observation, всеми её
  answer_ids/entities и дополнительно подтверждена кодовым детерминированным
  resolver. Текст модели сам по себе предупреждение не закрывает;
- verdict=revise допустим, только если есть хотя бы одно безопасное действие
  из разрешённого enum, оно ссылается на entity_name из entity_catalog и на
  существующие answer_id из audit_payload. Для revise также верни конкретную
  непустую annotation_guidance;
- не расширяй scope и не добавляй сущности. Не меняй raw-ответы, числители,
  знаменатели или готовые цифры вручную;
- attributed_to_target=false у подтверждённого сайтом standalone-продукта
  допустим без имени материнского бренда в raw и не отменяет портфельное
  попадание. require_target_attribution — сужение против ложноположительных
  совпадений, а не способ исправить такое ложное отрицание;
- evidence считается буквальным только при точном совпадении с raw, включая
  Markdown и регистр. Нельзя подтверждать acceptance_check нормализованной или
  искусственно склеенной цитатой. Если безопасное доказательство после ремонта
  отсутствует и сущность исключена из метрики, это корректный fail-closed
  результат, а не основание для второй такой же revise;
- если важную проблему нельзя исправить разрешёнными действиями, выбери block
  с пустыми policy_adjustments и annotation_guidance;
- не переноси critical/important anomaly в observation лишь для того, чтобы
  открыть gate: понижение допустимо только когда metric_contract доказывает,
  что находка не участвует в опубликованном расчёте.

Это единственная попытка восстановления. Пиши кратко и предметно по-русски.
""".strip()

CRITIC_RECOVERY_FINAL_SUFFIX = """
Это отдельная финальная проверка после одного ограниченного ремонта разметки,
который спланировал сильный оркестратор, но исполнил обычный разметчик. Решение
оркестратора не является доказательством. Самостоятельно сверь новый corpus,
аннотации и метрики с raw. Это последний gate: верни pass, только если все
critical/important проблемы устранены; иначе верни block. Новая revise и ещё
одна петля ремонта запрещены.
""".strip()


def _critic_iteration_contract(
    iteration: int,
    *,
    recovery_final: bool,
) -> tuple[int, bool]:
    """Validate the ordinary two rounds or the one recovery-only final gate."""

    recovery_iteration = (
        MAX_CRITIC_ITERATIONS + MAX_CRITIC_RECOVERY_FINAL_REVIEWS
    )
    if recovery_final:
        if iteration != recovery_iteration:
            raise ValueError(
                "Recovery final critic must use the dedicated final iteration"
            )
        return recovery_iteration, True
    if not 1 <= iteration <= MAX_CRITIC_ITERATIONS:
        raise ValueError("Critic iteration is outside the bounded loop")
    return MAX_CRITIC_ITERATIONS, False


def _normalize_review(parsed: dict[str, Any]) -> dict[str, Any]:
    # Preserve missing/null required values so the deterministic gate can
    # reject and repair them.  Converting null to an apparently valid empty
    # list made a malformed ``pass`` indistinguishable from a real audit.
    return dict(parsed)


def _stable_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _referenced_answer_ids(value: Any) -> set[int]:
    output: set[int] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "answer_ids" and isinstance(child, list):
                output.update(
                    answer_id
                    for answer_id in child
                    if isinstance(answer_id, int)
                    and not isinstance(answer_id, bool)
                )
            else:
                output.update(_referenced_answer_ids(child))
    elif isinstance(value, list):
        for child in value:
            output.update(_referenced_answer_ids(child))
    return output


def _critic_answer_index(answer: dict[str, Any]) -> dict[str, Any]:
    return {
        key: answer.get(key)
        for key in (
            "answer_id",
            "mode",
            "provider",
            "model",
            "prompt_id",
            "prompt_key",
            "scenario_role",
            "intent_class",
            "status",
            "annotation_state",
            "metric_eligible",
            "context_eligible",
            "metric_evidence_state",
            "raw_answer_sha256",
            "raw_answer_included",
            "raw_answer_char_count",
            "raw_answer_truncated",
            "raw_answer_omission_reason",
        )
    }


def _compact_repair_context(
    payload: dict[str, Any],
    incomplete_review: dict[str, Any],
) -> dict[str, Any]:
    """Keep repair evidence bounded without changing the audited payload."""

    answers = [
        answer
        for answer in payload.get("answers") or []
        if isinstance(answer, dict)
    ]
    referenced_ids = _referenced_answer_ids(
        [
            incomplete_review,
            payload.get("deterministic_warnings") or [],
        ]
    )
    affected = [
        answer
        for answer in answers
        if isinstance(answer.get("answer_id"), int)
        and answer.get("answer_id") in referenced_ids
    ]
    selected = affected[:CRITIC_REPAIR_MAX_AFFECTED_ANSWERS]
    selected_ids = {
        int(answer["answer_id"])
        for answer in selected
        if isinstance(answer.get("answer_id"), int)
    }
    affected_evidence: list[dict[str, Any]] = []
    for answer in selected:
        raw_answer = str(answer.get("raw_answer") or "")
        raw_included = answer.get("raw_answer_included")
        raw_char_count = answer.get("raw_answer_char_count")
        repair_raw_missing = bool(
            raw_included is False
            or (
                isinstance(raw_char_count, int)
                and raw_char_count > 0
                and not raw_answer
            )
        )
        affected_evidence.append(
            {
                **_critic_answer_index(answer),
                "scenario": answer.get("scenario"),
                "raw_answer": raw_answer[:CRITIC_REPAIR_RAW_CHAR_LIMIT],
                "repair_raw_missing": repair_raw_missing,
                "repair_raw_truncated": (
                    len(raw_answer) > CRITIC_REPAIR_RAW_CHAR_LIMIT
                ),
                "annotation": answer.get("annotation"),
            }
        )
    omitted_referenced_ids = sorted(referenced_ids - selected_ids)
    return {
        "audit_payload_sha256": _stable_sha256(payload),
        "site_profile": payload.get("site_profile"),
        "entity_catalog": payload.get("entity_catalog"),
        "metric_contract": payload.get("metric_contract"),
        "attribution_owner_aliases": payload.get(
            "attribution_owner_aliases"
        ),
        "entity_attribution_aliases": payload.get(
            "entity_attribution_aliases"
        ),
        "candidate_metrics": payload.get("candidate_metrics"),
        "deterministic_warnings": payload.get("deterministic_warnings") or [],
        "previous_policy_changes": payload.get("previous_policy_changes") or [],
        "answer_index": [_critic_answer_index(answer) for answer in answers],
        "affected_answer_evidence": affected_evidence,
        "evidence_limits": {
            "full_answer_count": len(answers),
            "referenced_answer_count": len(referenced_ids),
            "included_answer_count": len(affected_evidence),
            "max_included_answers": CRITIC_REPAIR_MAX_AFFECTED_ANSWERS,
            "raw_chars_per_answer": CRITIC_REPAIR_RAW_CHAR_LIMIT,
            "omitted_referenced_answer_ids": omitted_referenced_ids,
            "insufficient_evidence_requires_block": True,
        },
    }


def _critic_usage(
    usage: dict[str, Any],
    *,
    recovered_from: str | None = None,
) -> dict[str, Any]:
    output = dict(usage)
    transport = output.get("_aiv_transport")
    output["_aiv_critic_contract"] = {
        "version": CRITIC_TRANSPORT_CONTRACT_VERSION,
        "transport_status": (
            transport.get("status") if isinstance(transport, dict) else "unknown"
        ),
        "transport_output_complete": (
            transport.get("output_complete")
            if isinstance(transport, dict)
            else None
        ),
        "semantic_verdict_status": "pending_deterministic_validation",
        "semantic_validation_owner": "critic_gate",
        "recovered_from": recovered_from,
    }
    return output


def _merge_recovery_usage(
    primary_usage: dict[str, Any],
    repair_usage: dict[str, Any],
    *,
    recovered_from: str,
) -> dict[str, Any]:
    output = _critic_usage(repair_usage, recovered_from=recovered_from)
    output["_aiv_critic_attempts"] = [
        {"kind": "primary", "usage": primary_usage},
        {"kind": "compact_repair", "usage": repair_usage},
    ]
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "reasoning_tokens",
        "cost",
    ):
        values = [primary_usage.get(key), repair_usage.get(key)]
        if all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in values
        ):
            output[key] = values[0] + values[1]
    return output


def _incomplete_transport_review(
    error: OpenRouterResponseContractError,
) -> tuple[dict[str, Any], str]:
    transport = error.result.transport
    failure = (
        "output_limit"
        if isinstance(error, OpenRouterOutputLimitError)
        else "response_contract"
    )
    partial = error.result.text
    parsed_partial: dict[str, Any] | None = None
    try:
        candidate = json.loads(partial)
        if isinstance(candidate, dict):
            parsed_partial = candidate
    except (TypeError, json.JSONDecodeError):
        pass
    return (
        {
            "_transport_failure": failure,
            "_transport": transport,
            "_partial_response": partial[:CRITIC_PARTIAL_RESPONSE_CHAR_LIMIT],
            "_partial_response_sha256": hashlib.sha256(
                partial.encode("utf-8")
            ).hexdigest(),
            "_partial_response_chars": len(partial),
            "_partial_response_truncated": (
                len(partial) > CRITIC_PARTIAL_RESPONSE_CHAR_LIMIT
            ),
            "_parsed_partial_review": parsed_partial,
        },
        failure,
    )


def _transport_repair_may_pass(
    incomplete_review: dict[str, Any],
) -> bool:
    """A compact repair cannot invent a passing verdict from broken JSON."""

    partial = incomplete_review.get("_parsed_partial_review")
    if not isinstance(partial, dict) or partial.get("verdict") != "pass":
        return False
    anomalies = partial.get("anomalies")
    if not isinstance(anomalies, list):
        return False
    return not any(
        isinstance(anomaly, dict)
        and anomaly.get("severity") in {"critical", "important"}
        for anomaly in anomalies
    )


async def review_analysis(
    payload: dict[str, Any],
    *,
    iteration: int,
    recovery_final: bool = False,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Run one bounded independent critique of a candidate metric snapshot."""

    max_iterations, recovery_final = _critic_iteration_contract(
        iteration,
        recovery_final=recovery_final,
    )
    try:
        response = await chat(
            model=CRITIC_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        CRITIC_SYSTEM
                        + (
                            "\n\n" + CRITIC_RECOVERY_FINAL_SUFFIX
                            if recovery_final
                            else ""
                        )
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "iteration": iteration,
                            "max_iterations": max_iterations,
                            "recovery_final": recovery_final,
                            **payload,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            response_schema=CRITIC_SCHEMA,
            schema_name=f"aiv_analysis_critic_{iteration}",
            reasoning_effort=CRITIC_REASONING_EFFORT,
            max_tokens=CRITIC_MAX_TOKENS,
            temperature=0.1,
            retry_response_contract_errors=False,
        )
    except OpenRouterResponseContractError as exc:
        incomplete_review, failure = _incomplete_transport_review(exc)
        if recovery_final:
            # The final recovery gate has a one-call budget and must be a
            # fresh, complete primary verdict.  Even a partial JSON prefix
            # that says ``pass`` is not sufficient to publish.
            raise OpenRouterError(
                "Final recovery critic primary response was incomplete or "
                f"unparseable ({failure}); compact repair is forbidden"
            ) from exc
        repaired, raw_text, repair_usage = await repair_analysis_review(
            payload,
            incomplete_review,
            iteration=iteration,
            validation_errors=[
                "Primary critic transport completed but its structured response "
                f"was unusable ({failure}): {exc}"
            ],
            recovery_final=recovery_final,
        )
        if (
            repaired.get("verdict") == "pass"
            and not _transport_repair_may_pass(incomplete_review)
        ):
            raise OpenRouterError(
                "Compact critic repair cannot promote an unparseable or "
                "non-passing primary response to pass"
            )
        return (
            repaired,
            raw_text,
            _merge_recovery_usage(
                exc.result.usage,
                repair_usage,
                recovered_from=failure,
            ),
        )
    if not isinstance(response.parsed, dict):
        raise OpenRouterError("Analysis critic returned no structured verdict")
    parsed = _normalize_review(response.parsed)
    return parsed, response.text, _critic_usage(response.usage)


async def repair_analysis_review(
    payload: dict[str, Any],
    incomplete_review: dict[str, Any],
    *,
    iteration: int,
    validation_errors: list[str],
    recovery_final: bool = False,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Make one bounded attempt to repair an unusable critic decision.

    The immutable full payload stays bound by digest, while the repair request
    carries only metrics, catalog, an answer index and directly affected raw
    evidence. The repair model cannot acquire evidence or mutate measurements.
    """

    max_iterations, recovery_final = _critic_iteration_contract(
        iteration,
        recovery_final=recovery_final,
    )
    repair_context = _compact_repair_context(payload, incomplete_review)
    response = await chat(
        model=CRITIC_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    CRITIC_REPAIR_SYSTEM
                    + (
                        "\n\n" + CRITIC_RECOVERY_FINAL_SUFFIX
                        if recovery_final
                        else ""
                    )
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "iteration": iteration,
                        "max_iterations": max_iterations,
                        "recovery_final": recovery_final,
                        "repair_attempt": 1,
                        "max_repair_attempts": MAX_CRITIC_REPAIR_ATTEMPTS,
                        "validation_errors": validation_errors,
                        "incomplete_review": incomplete_review,
                        "audit_payload_sha256": _stable_sha256(payload),
                        "repair_context": repair_context,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        response_schema=CRITIC_SCHEMA,
        schema_name=f"aiv_analysis_critic_repair_{iteration}",
        reasoning_effort=CRITIC_REPAIR_REASONING_EFFORT,
        max_tokens=CRITIC_REPAIR_MAX_TOKENS,
        temperature=0.0,
        retry_response_contract_errors=False,
    )
    if not isinstance(response.parsed, dict):
        raise OpenRouterError(
            "Analysis critic repair returned no structured verdict"
        )
    repaired = _normalize_review(response.parsed)
    omitted_ids = repair_context["evidence_limits"][
        "omitted_referenced_answer_ids"
    ]
    truncated_ids = [
        evidence.get("answer_id")
        for evidence in repair_context["affected_answer_evidence"]
        if evidence.get("repair_raw_truncated") is True
    ]
    missing_ids = [
        evidence.get("answer_id")
        for evidence in repair_context["affected_answer_evidence"]
        if evidence.get("repair_raw_missing") is True
    ]
    if (
        omitted_ids or truncated_ids or missing_ids
    ) and repaired.get("verdict") != "block":
        raise OpenRouterError(
            "Analysis critic repair used incomplete affected evidence; "
            "only block is safe"
        )
    return (
        repaired,
        response.text,
        _critic_usage(response.usage),
    )
