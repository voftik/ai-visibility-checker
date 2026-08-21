from __future__ import annotations

import json
from typing import Any

from app.config import settings
from app.services.openrouter import OpenRouterError, chat

CRITIC_VERSION = "aiv-analysis-critic-v15"
MAX_CRITIC_ITERATIONS = 2
MAX_CRITIC_REPAIR_ATTEMPTS = 1
CRITIC_MODEL = settings.OPENROUTER_CRITIC_MODEL

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
профиль исследуемого сайта, каталог сущностей, сценарии, полные raw-ответы,
разметка и уже рассчитанные срезы.

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
- если хотя бы у одного ответа raw_answer_truncated=true, полного контекста
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
и не придумывай новые факты. Сверь исходное решение с audit_payload и верни
самодостаточное исправленное решение в той же строгой схеме.

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


def _normalize_review(parsed: dict[str, Any]) -> dict[str, Any]:
    # Preserve missing/null required values so the deterministic gate can
    # reject and repair them.  Converting null to an apparently valid empty
    # list made a malformed ``pass`` indistinguishable from a real audit.
    return dict(parsed)


async def review_analysis(
    payload: dict[str, Any],
    *,
    iteration: int,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Run one bounded independent critique of a candidate metric snapshot."""

    if not 1 <= iteration <= MAX_CRITIC_ITERATIONS:
        raise ValueError("Critic iteration is outside the bounded loop")
    response = await chat(
        model=CRITIC_MODEL,
        messages=[
            {"role": "system", "content": CRITIC_SYSTEM},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "iteration": iteration,
                        "max_iterations": MAX_CRITIC_ITERATIONS,
                        **payload,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        response_schema=CRITIC_SCHEMA,
        schema_name=f"aiv_analysis_critic_{iteration}",
        reasoning_effort="high",
        max_tokens=12_000,
        temperature=0.1,
    )
    if not isinstance(response.parsed, dict):
        raise OpenRouterError("Analysis critic returned no structured verdict")
    parsed = _normalize_review(response.parsed)
    return parsed, response.text, response.usage


async def repair_analysis_review(
    payload: dict[str, Any],
    incomplete_review: dict[str, Any],
    *,
    iteration: int,
    validation_errors: list[str],
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Make one bounded attempt to repair an unusable critic decision.

    The full audit payload is intentionally preserved. The repair model may
    correct the decision contract, but it cannot acquire new evidence or
    mutate any measured value.
    """

    if not 1 <= iteration <= MAX_CRITIC_ITERATIONS:
        raise ValueError("Critic iteration is outside the bounded loop")
    response = await chat(
        model=CRITIC_MODEL,
        messages=[
            {"role": "system", "content": CRITIC_REPAIR_SYSTEM},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "iteration": iteration,
                        "repair_attempt": 1,
                        "max_repair_attempts": MAX_CRITIC_REPAIR_ATTEMPTS,
                        "validation_errors": validation_errors,
                        "incomplete_review": incomplete_review,
                        "audit_payload": payload,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        response_schema=CRITIC_SCHEMA,
        schema_name=f"aiv_analysis_critic_repair_{iteration}",
        reasoning_effort="high",
        max_tokens=12_000,
        temperature=0.0,
    )
    if not isinstance(response.parsed, dict):
        raise OpenRouterError(
            "Analysis critic repair returned no structured verdict"
        )
    return (
        _normalize_review(response.parsed),
        response.text,
        response.usage,
    )
