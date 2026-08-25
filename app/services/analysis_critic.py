from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import Awaitable, Callable
from typing import Any

from jsonschema import Draft202012Validator

from app.config import settings
from app.services.long_response import (
    DEFAULT_CONTEXT_OVERLAP_CHARS,
    TextUnit,
    split_lossless_text,
    verify_units,
)
from app.services.openrouter import (
    AuditCheckpoint,
    OpenRouterError,
    OpenRouterOutputLimitError,
    OpenRouterResponseContractError,
    OutputTokenPolicy,
    WebSearchPolicy,
    _apply_model_output_headroom,
    chat,
    model_output_envelope,
    restore_completed_chat_provider_event,
    web_request_policy,
)

CRITIC_VERSION = "aiv-analysis-critic-v24"
CRITIC_TRANSPORT_CONTRACT_VERSION = "aiv-analysis-critic-transport-v1"
CRITIC_MAP_REDUCE_VERSION = "aiv-analysis-critic-map-reduce-v4"
CRITIC_CALL_AUDIT_VERSION = "aiv-analysis-critic-call-audit-v1"
MAX_CRITIC_ITERATIONS = 2
MAX_CRITIC_RECOVERY_FINAL_REVIEWS = 1
MAX_CRITIC_REPAIR_ATTEMPTS = 1
CRITIC_MODEL = settings.OPENROUTER_CRITIC_MODEL
CRITIC_REASONING_EFFORT = "medium"
CRITIC_REPAIR_REASONING_EFFORT = "low"

# UTF-8 bytes are used as a deliberately conservative upper bound for tokens.
# This is a per-request window, never a corpus cap: an arbitrary number of
# complete answers can be assigned to an arbitrary number of leaf calls.
CRITIC_SMALL_SINGLE_CALL_BYTES = 64_000
CRITIC_MAP_CONCURRENCY = 3
CRITIC_REQUEST_PROTOCOL_TOKEN_UPPER_BOUND = 256

CriticCallAuditSink = Callable[[dict[str, Any]], Awaitable[None]]
TransportResumeLookup = Callable[
    [str],
    Awaitable[dict[str, Any] | None],
]


def _critic_transport_key(
    *,
    messages: list[dict[str, Any]],
    schema_name: str,
    reasoning_effort: str,
    temperature: float,
) -> str:
    return _stable_sha256(
        {
            "version": CRITIC_TRANSPORT_CONTRACT_VERSION,
            "model": CRITIC_MODEL,
            "messages": messages,
            "schema_name": schema_name,
            "schema_sha256": _stable_sha256(CRITIC_SCHEMA),
            "reasoning_effort": reasoning_effort,
            "temperature": temperature,
            "web_policy": WebSearchPolicy.FORBIDDEN.value,
            "output_token_policy": OutputTokenPolicy.MODEL_MAX.value,
        }
    )


async def _critic_atomic_chat(
    *,
    messages: list[dict[str, Any]],
    schema_name: str,
    reasoning_effort: str,
    temperature: float,
    transport_audit_checkpoint: AuditCheckpoint | None,
    transport_resume_lookup: TransportResumeLookup | None,
) -> Any:
    """Run or restore one atomic critic verdict without a rebilling gap."""

    transport_key = _critic_transport_key(
        messages=messages,
        schema_name=schema_name,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
    )
    if transport_resume_lookup is not None:
        cached = await asyncio.shield(transport_resume_lookup(transport_key))
        if cached is not None:
            if not isinstance(cached, dict) or (
                cached.get("document_id") != transport_key
            ):
                raise OpenRouterError(
                    "Critic transport receipt identity mismatch"
                )
            return restore_completed_chat_provider_event(
                cached,
                model=CRITIC_MODEL,
                messages=messages,
                web_policy=WebSearchPolicy.FORBIDDEN,
                temperature=temperature,
                response_schema=CRITIC_SCHEMA,
                schema_name=schema_name,
                reasoning_effort=reasoning_effort,
            )
    transport_kwargs: dict[str, Any] = {}
    if transport_audit_checkpoint is not None:
        transport_kwargs = {
            "audit_checkpoint": transport_audit_checkpoint,
            "audit_context": {"document_id": transport_key},
        }
    return await chat(
        model=CRITIC_MODEL,
        messages=messages,
        response_schema=CRITIC_SCHEMA,
        schema_name=schema_name,
        reasoning_effort=reasoning_effort,
        output_token_policy=OutputTokenPolicy.MODEL_MAX,
        temperature=temperature,
        retry_response_contract_errors=False,
        retry_transport_errors=False,
        **transport_kwargs,
    )

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
raw-ответов с SHA, разметка и уже рассчитанные срезы. Для каждой непустой
сохранённой строки передан весь фактически полученный raw-текст без
посимвольной обрезки. raw_answer_manifest фиксирует длину, digest и все
lossless-части, использованные предыдущими этапами. Если
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
- metric_evidence_state=provider_limited_prefix означает, что сохранён весь
  фактически выданный провайдером префикс, но модель достигла своего
  физического предела. Такая строка не входит в знаменатели и не может
  подтверждать отсутствие бренда или сущности. Разрешено учитывать только
  буквальные положительные факты, уже присутствующие в сохранённом тексте;
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

CRITIC_REPAIR_FRAGMENT_SUFFIX = """

Это один lossless-фрагмент raw-ответа, который нужен для восстановления
контракта critic-решения. Поле repair_fragment задаёт исходный answer_id,
непересекающийся core и контекстное окно с overlap. Делай вывод только по
этому окну. Находка принадлежит фрагменту, в core которого начинается её
решающее буквальное evidence; overlap не создаёт второй голос. Raw других
затронутых ответов намеренно обрабатывается соседними leaf-вызовами, поэтому
его отсутствие здесь не означает потерю данных. Не ослабляй исходную важную
находку и не делай глобальный pass только по одной части.
""".strip()

CRITIC_REPAIR_REDUCE_SYSTEM = """
Ты reducer единственной попытки восстановления critic-решения AI visibility.
Каждый leaf уже проверил назначенный ему непересекающийся code-owned core.
Собери решение в исходной строгой схеме, не придумывая новых raw-фактов.
Lineage и полнота покрытия принадлежат коду, а не модели.

Нельзя снижать verdict частей: block остаётся block, иначе revise задаёт
минимум revise. Сохрани все critical/important anomalies, policy_adjustments,
answer_ids и entities. Точные overlap-дубли можно объединить, но материальную
находку нельзя удалить. При pass policy_adjustments=[],
annotation_guidance="" и есть конкретная acceptance_check.
""".strip()

CRITIC_RECOVERY_FINAL_SUFFIX = """
Это отдельная финальная проверка после одного ограниченного ремонта разметки,
который спланировал сильный оркестратор, но исполнил обычный разметчик. Решение
оркестратора не является доказательством. Самостоятельно сверь новый corpus,
аннотации и метрики с raw. Это последний gate: верни pass, только если все
critical/important проблемы устранены; иначе верни block. Новая revise и ещё
одна петля ремонта запрещены.
""".strip()

CRITIC_MAP_LEAF_SUFFIX = """

Это одна code-owned часть полного raw-корпуса. Если общий контекст помещается,
поля site_profile, entity_catalog, metric_contract, candidate_metrics и
deterministic_warnings переданы напрямую. Если он был больше окна, этот leaf
вызывается как серия joint shards: shared_context_receipt фиксирует результат
context-only проверки, а shared_context_facts передаёт точный факт либо его
lossless semantic-overlap фрагмент. Полный cross-product всех фактов и
answer/query leaves проверяет код; receipt или hash не заменяет содержание.
Полный контекстный verdict отдельно войдёт в финальный code-owned floor.
Полный индекс корпуса закреплён count и SHA в
complete_answer_index_manifest; в assigned_answer_index находятся только
строки этой части. Полный индекс и манифесты получит reducer, а code-owned
provenance проверит покрытие. Raw-текст есть только у answer_id из
critic_map_partition.assigned_answer_ids. Каждый raw-ответ попадает ровно в
одну часть и всегда целиком. Не считай отсутствие raw других частей потерей
данных и не делай по ним отрицательных выводов. Проверь все назначенные этой
части ответы и верни локальный verdict: любая найденная critical/important
проблема должна остаться critical/important, чтобы финальный reducer не мог
её потерять.
""".strip()

CRITIC_REDUCE_SYSTEM = """
Ты финальный reducer независимого критика AI visibility. Raw-ответы уже
проверены в непересекающихся code-owned частях. Тебе переданы полный индекс и
SHA-манифесты корпуса, общий контекст либо его фактический bounded digest после
lossless context-reduce, а также все answer leaf-verdict. Полный context
verdict дополнительно защищён code-owned floor. Собери одно решение в исходной
строгой схеме.

Нельзя ослаблять решения частей: если хотя бы одна часть выбрала block,
итоговый verdict — block; иначе при хотя бы одном revise итог не ниже revise.
Сохрани все critical/important anomalies, policy_adjustments, связанные
answer_ids и entities. Можно объединить точные дубли, уточнить summary и
добавить межчастевую аномалию, но нельзя удалять материальную находку. При
итоговом pass policy_adjustments=[], annotation_guidance="" и есть хотя бы
одна конкретная acceptance_check. Не придумывай raw-факты: reducer получает
только решения частей и provenance.
""".strip()

CRITIC_FRAGMENT_SUFFIX = """

Этот leaf содержит semantic-overlap фрагмент одного слишком длинного
raw-ответа. Поле raw_answer — контекстное окно; его точный непересекающийся
core задан смещениями critic_fragment.core_start_in_context и
core_end_in_context. Контекст до и после core нужен только для понимания связи
на границе. Он не является второй копией evidence. Владельцем находки считается
фрагмент, core которого содержит первый символ решающего буквального evidence.
Если evidence началось в левом overlap, не дублируй находку соседнего core.
Аннотация относится ко всему исходному ответу и поэтому её _answer_sha256
сверяется с critic_fragment.source_sha256, а не с digest контекстного окна.
Сошлись на исходный answer_id; unit lineage проверяет код.
""".strip()

CRITIC_ANSWER_REDUCE_SYSTEM = """
Ты reducer всех semantic-overlap фрагментов одного raw-ответа в независимом
аудите AI visibility. Полный ответ покрыт непересекающимися core-диапазонами;
overlap только сохраняет смысл на границе и не удваивает evidence. Собери один
verdict в исходной строгой схеме. Не ослабляй части: любой block остаётся
block, иначе любой revise задаёт минимум revise. Сохрани все
critical/important anomalies, answer_ids, entities и policy_adjustments.
Точные дубли overlap можно объединить, но материальную находку нельзя удалить.
""".strip()

CRITIC_SHARED_CONTEXT_LEAF_SUFFIX = """

Это одна lossless-часть общего контекста исследования, а не raw-ответ
пользователю. Поле shared_context_json_fragment содержит контекстное окно
канонического JSON; непересекающийся core и overlap заданы в
critic_shared_context_partition. Проверь факты о сайте, сущностях, правилах,
метриках и deterministic warnings только в назначенном core. Не делай выводов
об отсутствии raw-ответов: они будут проверены после сведения общего
контекста. Любую material-проблему сохрани в verdict, чтобы reducer не мог её
потерять.
""".strip()

CRITIC_SHARED_CONTEXT_REDUCE_SYSTEM = """
Ты reducer lossless-частей общего контекста исследования AI visibility.
Каждый leaf проверил один непересекающийся code-owned core канонического JSON.
Собери компактный рабочий digest правил, сущностей, метрик и material-находок
в исходной critic-схеме. Не ослабляй verdict частей и не удаляй
critical/important findings. Этот digest будет передан каждому answer leaf,
а полные leaf-verdict также останутся в финальном code-owned verdict floor.
""".strip()

CRITIC_CONTEXT_JOIN_SUFFIX = """

Это code-owned joint leaf: назначенные raw-ответы или их lossless-фрагменты
проверяются вместе с точными единицами общего контекста. Поле
critic_context_binding задаёт неизменяемые answer/query IDs, полный inventory
и непересекающееся покрытие context_unit_ids. Поле shared_context_facts
содержит либо полный атомарный факт, либо точный semantic-overlap фрагмент его
канонического JSON с core-диапазоном. Фрагмент нельзя заменять догадкой по
хешу. Любая связь факта с ответом, запросом, аннотацией или метрикой должна
проверяться только по переданному содержимому. Соседние joint leaves покрывают
остальные факты; отсутствие их текста здесь не означает отсутствие факта.
Lineage и полнота cross-product проверяются кодом до финального reducer.
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


def _critic_logical_call_descriptor(
    *,
    iteration: int,
    kind: str,
    index: int,
    schema_name: str,
    call_input: dict[str, Any],
    lineage: dict[str, Any],
) -> dict[str, Any]:
    """Build the durable identity of one paid logical critic call.

    ``attempt_id`` used to be random, so a process restart could not discover
    that an identical leaf had already completed.  The identity below includes
    every value that may change semantics or structured parsing.  A storage
    adapter may keep unique append-only transport receipts separately, while
    completed logical results are addressed by this digest.
    """

    identity = {
        "audit_version": CRITIC_CALL_AUDIT_VERSION,
        "critic_version": CRITIC_VERSION,
        "map_reduce_version": CRITIC_MAP_REDUCE_VERSION,
        "model": CRITIC_MODEL,
        "iteration": iteration,
        "kind": kind,
        "index": index,
        "schema_name": schema_name,
        "schema_sha256": _stable_sha256(CRITIC_SCHEMA),
        "input_sha256": _stable_sha256(call_input),
        "lineage_sha256": _stable_sha256(lineage),
    }
    return {
        **identity,
        "logical_call_key": _stable_sha256(identity),
    }


def _critic_attempt_id(descriptor: dict[str, Any]) -> str:
    logical_key = str(descriptor.get("logical_call_key") or "")
    if len(logical_key) != 64:
        raise OpenRouterError("Critic logical call has no stable SHA-256 key")
    return logical_key[:32]


async def _lookup_completed_critic_call(
    audit_sink: CriticCallAuditSink | None,
    descriptor: dict[str, Any],
) -> tuple[dict[str, Any], str, dict[str, Any]] | None:
    """Reuse a completed paid result exposed by a durable audit adapter.

    Existing write-only callables remain compatible.  A resumable adapter adds
    an async ``lookup_completed(descriptor)`` method and returns the previously
    persisted audit event.  Cache data is fail-closed: a digest/schema mismatch
    is never silently treated as a hit.
    """

    if audit_sink is None:
        return None
    lookup = getattr(audit_sink, "lookup_completed", None)
    if not callable(lookup):
        return None
    cached = await asyncio.shield(lookup(_json_safe(descriptor)))
    if cached is None:
        return None
    if not isinstance(cached, dict):
        raise OpenRouterError("Critic audit lookup returned a non-object")
    if cached.get("status") != "completed":
        return None
    expected = {
        "version": CRITIC_CALL_AUDIT_VERSION,
        "critic_version": CRITIC_VERSION,
        "model": CRITIC_MODEL,
        "logical_call_key": descriptor["logical_call_key"],
        "schema_name": descriptor["schema_name"],
        "input_sha256": descriptor["input_sha256"],
        "lineage_sha256": descriptor["lineage_sha256"],
    }
    mismatches = {
        key: {"expected": value, "actual": cached.get(key)}
        for key, value in expected.items()
        if cached.get(key) != value
    }
    if mismatches:
        raise OpenRouterError(
            "Critic audit cache identity mismatch: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )
    review = cached.get("output")
    raw_text = cached.get("raw_text")
    usage = cached.get("usage")
    if not isinstance(review, dict) or not isinstance(raw_text, str):
        raise OpenRouterError(
            "Completed critic audit cache has no structured output/raw text"
        )
    if not isinstance(usage, dict):
        raise OpenRouterError("Completed critic audit cache has invalid usage")
    if (
        cached.get("output_sha256") != _stable_sha256(review)
        or cached.get("raw_response_sha256")
        != hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
        or cached.get("raw_response_chars") != len(raw_text)
    ):
        raise OpenRouterError(
            "Completed critic audit cache content digest is missing or tampered"
        )
    validated = _validated_review_schema(
        review,
        owner="Resumed critic audit result",
    )
    resumed_usage = dict(usage)
    resumed_usage["_aiv_critic_resume"] = {
        "reused": True,
        "logical_call_key": descriptor["logical_call_key"],
        "attempt_id": _critic_attempt_id(descriptor),
    }
    return validated, raw_text, _json_safe(resumed_usage)


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
    """Build repair evidence without shortening any selected raw answer."""

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
    selected = affected
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
                "raw_answer": raw_answer,
                "repair_raw_missing": repair_raw_missing,
                "repair_raw_truncated": False,
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
            "max_included_answers": None,
            "raw_chars_per_answer": None,
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
            "_partial_response": partial,
            "_partial_response_sha256": hashlib.sha256(
                partial.encode("utf-8")
            ).hexdigest(),
            "_partial_response_chars": len(partial),
            "_partial_response_truncated": False,
            "_parsed_partial_review": parsed_partial,
        },
        failure,
    )


def _output_limit_fail_closed_result(
    incomplete_review: dict[str, Any],
    *,
    primary_usage: dict[str, Any],
    decision_shard_attempts: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Block without granting semantic authority to a limited prefix.

    A provider may return syntactically valid JSON and still declare the
    response output-limited.  In that state neither the listed anomalies nor
    the policy adjustments prove that the model finished enumerating its
    decision.  The complete received prefix remains in the raw audit ledger,
    but no field from it is copied into the executable review.
    """

    partial_sha256 = str(
        incomplete_review.get("_partial_response_sha256") or ""
    )
    partial_chars = incomplete_review.get("_partial_response_chars")
    review = _validated_review_schema(
        {
            "verdict": "block",
            "summary": (
                "Независимый критик достиг физического лимита ответа. "
                "Полученный префикс сохранён для аудита, но не является "
                "полным решением и не может менять правила анализа."
            ),
            "anomalies": [],
            "policy_adjustments": [],
            "annotation_guidance": "",
            "acceptance_checks": [
                "Повторить critic-проверку в независимых decision-shards с "
                "явным полным покрытием находок и policy adjustments."
            ],
        },
        owner="Output-limited critic fail-closed decision",
    )
    coverage = {
        "semantic_authority": "none",
        "anomalies_complete": False,
        "policy_adjustments_complete": False,
        "acceptance_checks_complete": False,
        "partial_response_sha256": partial_sha256,
        "partial_response_chars": partial_chars,
        "partial_response_truncated_by_application": False,
    }
    attempts = list(decision_shard_attempts or [])
    raw_text = json.dumps(
        {
            "version": CRITIC_TRANSPORT_CONTRACT_VERSION,
            "status": "fail_closed",
            "failure": "output_limit",
            "coverage": coverage,
            "primary_partial": incomplete_review,
            "decision_shard_attempts": attempts,
            "final_review_sha256": _stable_sha256(review),
        },
        ensure_ascii=False,
    )
    aggregate_usage = dict(primary_usage)
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "reasoning_tokens",
        "cost",
    ):
        values = [
            value
            for value in [
                primary_usage.get(key),
                *[
                    item.get("usage", {}).get(key)
                    for item in attempts
                    if isinstance(item, dict)
                    and isinstance(item.get("usage"), dict)
                ],
            ]
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        if values:
            aggregate_usage[key] = sum(values)
    usage = _critic_usage(
        aggregate_usage,
        recovered_from="output_limit_fail_closed",
    )
    usage["_aiv_critic_attempts"] = [
        {
            "kind": "primary_output_limited",
            "usage": dict(primary_usage),
        },
        *[
            {
                "kind": "output_limit_decision_shard",
                "decision": item.get("decision"),
                "status": item.get("status"),
                "usage": copy_usage,
            }
            for item in attempts
            if isinstance(item, dict)
            for copy_usage in [dict(item.get("usage") or {})]
        ],
    ]
    usage["_aiv_critic_output_limit"] = {
        "version": CRITIC_TRANSPORT_CONTRACT_VERSION,
        "status": "fail_closed",
        **coverage,
    }
    return review, raw_text, usage


_CRITIC_OUTPUT_LIMIT_DECISIONS: tuple[str, ...] = (
    "anomalies",
    "policy_adjustments",
    "conclusion",
)


async def _recover_output_limited_review_by_decision(
    payload: dict[str, Any],
    incomplete_review: dict[str, Any],
    *,
    iteration: int,
    max_iterations: int,
    recovery_final: bool,
    schema_name: str,
    map_leaf: bool,
    fragment_leaf: bool,
    shared_context_leaf: bool,
    context_join_leaf: bool,
    validate_schema: bool,
    primary_usage: dict[str, Any],
    transport_audit_checkpoint: AuditCheckpoint | None,
    transport_resume_lookup: TransportResumeLookup | None,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Recover a complete verdict through three bounded decision shards.

    The limited prefix has no semantic authority. Each shard independently
    rereads the same evidence but may emit only one decision class, sharply
    bounding physical output. All three must complete and satisfy their
    section contract before the deterministic union becomes authoritative;
    otherwise the result remains fail-closed.
    """

    shard_reviews: dict[str, dict[str, Any]] = {}
    attempt_rows: list[dict[str, Any]] = []
    accumulated_usage: dict[str, float] = {
        key: value
        for key in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "reasoning_tokens",
            "cost",
        )
        if isinstance((value := primary_usage.get(key)), (int, float))
        and not isinstance(value, bool)
    }
    instructions = {
        "anomalies": (
            "Верни полный список anomalies. policy_adjustments и "
            "acceptance_checks должны быть пусты, annotation_guidance=''."
        ),
        "policy_adjustments": (
            "Верни полный список policy_adjustments. anomalies и "
            "acceptance_checks должны быть пусты, annotation_guidance=''."
        ),
        "conclusion": (
            "Верни итоговый verdict, summary, полный annotation_guidance и "
            "acceptance_checks. anomalies и policy_adjustments должны быть пусты."
        ),
    }
    for decision_index, decision in enumerate(
        _CRITIC_OUTPUT_LIMIT_DECISIONS
    ):
        shard_payload = {
            **payload,
            "critic_output_limit_recovery": {
                "version": CRITIC_MAP_REDUCE_VERSION,
                "primary_partial_sha256": incomplete_review.get(
                    "_partial_response_sha256"
                ),
                "decision": decision,
                "decision_index": decision_index,
                "decision_count": len(_CRITIC_OUTPUT_LIMIT_DECISIONS),
                "instruction": instructions[decision],
                "limited_prefix_has_semantic_authority": False,
            },
        }
        messages = [
            {
                "role": "system",
                "content": _critic_system_prompt(
                    recovery_final=recovery_final,
                    map_leaf=map_leaf,
                    fragment_leaf=fragment_leaf,
                    shared_context_leaf=shared_context_leaf,
                    context_join_leaf=context_join_leaf,
                )
                + "\n\n"
                + instructions[decision],
            },
            {
                "role": "user",
                "content": json.dumps(
                    _critic_command(
                        shard_payload,
                        iteration=iteration,
                        max_iterations=max_iterations,
                        recovery_final=recovery_final,
                    ),
                    ensure_ascii=False,
                ),
            },
        ]
        try:
            response = await _critic_atomic_chat(
                messages=messages,
                schema_name=f"{schema_name}_output_{decision_index}",
                reasoning_effort=CRITIC_REASONING_EFFORT,
                temperature=0.1,
                transport_audit_checkpoint=transport_audit_checkpoint,
                transport_resume_lookup=transport_resume_lookup,
            )
            if not isinstance(response.parsed, dict):
                raise OpenRouterError(
                    f"Critic output decision shard {decision} has no verdict"
                )
            review = _validated_review_schema(
                response.parsed,
                owner=f"Critic output decision shard {decision}",
            )
            forbidden = {
                "anomalies": (
                    review.get("policy_adjustments"),
                    review.get("acceptance_checks"),
                    review.get("annotation_guidance"),
                ),
                "policy_adjustments": (
                    review.get("anomalies"),
                    review.get("acceptance_checks"),
                    review.get("annotation_guidance"),
                ),
                "conclusion": (
                    review.get("anomalies"),
                    review.get("policy_adjustments"),
                ),
            }[decision]
            if any(bool(value) for value in forbidden):
                raise OpenRouterError(
                    f"Critic output decision shard {decision} crossed sections"
                )
            shard_reviews[decision] = review
            usage = _critic_usage(response.usage)
            for key in (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "reasoning_tokens",
                "cost",
            ):
                value = usage.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    accumulated_usage[key] = accumulated_usage.get(key, 0) + value
            attempt_rows.append(
                {
                    "decision": decision,
                    "status": "completed",
                    "raw_text": str(response.text or ""),
                    "raw_text_sha256": hashlib.sha256(
                        str(response.text or "").encode("utf-8")
                    ).hexdigest(),
                    "review_sha256": _stable_sha256(review),
                    "usage": usage,
                }
            )
        except (OpenRouterResponseContractError, OpenRouterError) as exc:
            result = getattr(exc, "result", None)
            partial = str(getattr(result, "text", "") or "")
            attempt_rows.append(
                {
                    "decision": decision,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "raw_text": partial,
                    "raw_text_sha256": hashlib.sha256(
                        partial.encode("utf-8")
                    ).hexdigest(),
                    "usage": dict(getattr(result, "usage", {}) or {}),
                }
            )
            # Continue through the finite decision plan so every section gets
            # an independent durable attempt/receipt. No partial subset gains
            # authority; one failed shard still makes the final result block.
            continue

    if set(shard_reviews) != set(_CRITIC_OUTPUT_LIMIT_DECISIONS):
        return _output_limit_fail_closed_result(
            incomplete_review,
            primary_usage=primary_usage,
            decision_shard_attempts=attempt_rows,
        )

    anomalies = shard_reviews["anomalies"]
    policies = shard_reviews["policy_adjustments"]
    conclusion = shard_reviews["conclusion"]
    rank = {"pass": 0, "revise": 1, "block": 2}
    verdict = max(
        (review["verdict"] for review in shard_reviews.values()),
        key=lambda value: rank.get(str(value), 2),
    )
    merged = _validated_review_schema(
        {
            "verdict": verdict,
            "summary": " ".join(
                dict.fromkeys(
                    str(review.get("summary") or "").strip()
                    for review in shard_reviews.values()
                    if str(review.get("summary") or "").strip()
                )
            ),
            "anomalies": anomalies["anomalies"],
            "policy_adjustments": policies["policy_adjustments"],
            "annotation_guidance": conclusion["annotation_guidance"],
            "acceptance_checks": conclusion["acceptance_checks"],
        },
        owner="Critic output-limit decision-shard union",
    )
    raw_text = json.dumps(
        {
            "version": CRITIC_TRANSPORT_CONTRACT_VERSION,
            "status": "recovered_by_complete_decision_shards",
            "primary_partial": incomplete_review,
            "decision_shards": attempt_rows,
            "final_review_sha256": _stable_sha256(merged),
        },
        ensure_ascii=False,
    )
    usage = _critic_usage(
        {**accumulated_usage},
        recovered_from="output_limit_decision_shards",
    )
    usage["_aiv_critic_attempts"] = [
        {"kind": "primary_output_limited", "usage": dict(primary_usage)},
        *[
            {
                "kind": "output_limit_decision_shard",
                "decision": row["decision"],
                "status": row["status"],
                "usage": dict(row.get("usage") or {}),
            }
            for row in attempt_rows
        ],
    ]
    usage["_aiv_critic_output_limit"] = {
        "version": CRITIC_TRANSPORT_CONTRACT_VERSION,
        "status": "recovered_by_complete_decision_shards",
        "semantic_authority": "complete_decision_shard_union",
        "decision_count": len(attempt_rows),
        "decision_shards_complete": True,
    }
    return (
        merged if validate_schema else _normalize_review(merged),
        raw_text,
        usage,
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


def _json_safe(value: Any) -> Any:
    """Return a value that can be persisted in a JSON database column."""

    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(child) for child in value]
    return str(value)


def _critic_system_prompt(
    *,
    recovery_final: bool,
    map_leaf: bool = False,
    fragment_leaf: bool = False,
    shared_context_leaf: bool = False,
    context_join_leaf: bool = False,
) -> str:
    parts = [CRITIC_SYSTEM]
    if recovery_final:
        parts.append(CRITIC_RECOVERY_FINAL_SUFFIX)
    if map_leaf:
        parts.append(CRITIC_MAP_LEAF_SUFFIX)
    if fragment_leaf:
        parts.append(CRITIC_FRAGMENT_SUFFIX)
    if shared_context_leaf:
        parts.append(CRITIC_SHARED_CONTEXT_LEAF_SUFFIX)
    if context_join_leaf:
        parts.append(CRITIC_CONTEXT_JOIN_SUFFIX)
    return "\n\n".join(parts)


def _critic_command(
    payload: dict[str, Any],
    *,
    iteration: int,
    max_iterations: int,
    recovery_final: bool,
) -> dict[str, Any]:
    return {
        "iteration": iteration,
        "max_iterations": max_iterations,
        "recovery_final": recovery_final,
        **payload,
    }


def _json_utf8_bytes(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False).encode("utf-8"))


def _critic_request_utf8_bytes(
    payload: dict[str, Any],
    *,
    iteration: int,
    max_iterations: int,
    recovery_final: bool,
    map_leaf: bool = False,
    fragment_leaf: bool = False,
    shared_context_leaf: bool = False,
    context_join_leaf: bool = False,
    schema_name: str = "aiv_analysis_critic_preflight",
    context_envelope: dict[str, Any] | None = None,
) -> int:
    if isinstance(context_envelope, dict):
        return int(
            _critic_physical_request_preflight(
                payload,
                iteration=iteration,
                max_iterations=max_iterations,
                recovery_final=recovery_final,
                schema_name=schema_name,
                context_envelope=context_envelope,
                map_leaf=map_leaf,
                fragment_leaf=fragment_leaf,
                shared_context_leaf=shared_context_leaf,
                context_join_leaf=context_join_leaf,
            )["request_utf8_bytes"]
        )
    messages = [
        {
            "role": "system",
            "content": _critic_system_prompt(
                recovery_final=recovery_final,
                map_leaf=map_leaf,
                fragment_leaf=fragment_leaf,
                shared_context_leaf=shared_context_leaf,
                context_join_leaf=context_join_leaf,
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                _critic_command(
                    payload,
                    iteration=iteration,
                    max_iterations=max_iterations,
                    recovery_final=recovery_final,
                ),
                ensure_ascii=False,
            ),
        },
    ]
    return _json_utf8_bytes(messages)


def _critic_physical_request_preflight(
    payload: dict[str, Any],
    *,
    iteration: int,
    max_iterations: int,
    recovery_final: bool,
    schema_name: str,
    context_envelope: dict[str, Any],
    map_leaf: bool = False,
    fragment_leaf: bool = False,
    shared_context_leaf: bool = False,
    context_join_leaf: bool = False,
    temperature: float = 0.1,
) -> dict[str, Any]:
    """Preflight the exact first physical POST body used by ``chat``.

    The same model, messages, forbidden-web fields, reasoning contract,
    response schema, temperature and model-max headroom algorithm are used by
    :func:`app.services.openrouter.chat`.  The returned digest therefore binds
    planning to the complete physical request rather than only its messages.
    """

    messages = [
        {
            "role": "system",
            "content": _critic_system_prompt(
                recovery_final=recovery_final,
                map_leaf=map_leaf,
                fragment_leaf=fragment_leaf,
                shared_context_leaf=shared_context_leaf,
                context_join_leaf=context_join_leaf,
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                _critic_command(
                    payload,
                    iteration=iteration,
                    max_iterations=max_iterations,
                    recovery_final=recovery_final,
                ),
                ensure_ascii=False,
            ),
        },
    ]
    physical: dict[str, Any] = {
        "model": CRITIC_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    policy_fields, _policy = web_request_policy(
        model=CRITIC_MODEL,
        policy="forbidden",
    )
    physical.update(policy_fields)
    physical["reasoning"] = {
        "effort": CRITIC_REASONING_EFFORT,
        "exclude": True,
    }
    physical["response_format"] = {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "strict": True,
            "schema": CRITIC_SCHEMA,
        },
    }
    requested_max = context_envelope.get("max_completion_tokens")
    if not isinstance(requested_max, int) or isinstance(requested_max, bool):
        requested_max = context_envelope.get("reserved_output_tokens")
    if not isinstance(requested_max, int) or isinstance(requested_max, bool):
        requested_max = None
    try:
        resolved_max, resolved_envelope = _apply_model_output_headroom(
            payload=physical,
            output_envelope=context_envelope,
            requested_max=requested_max,
        )
    except OpenRouterError as exc:
        if "no completion headroom" not in str(exc):
            raise
        # Admission failure is a normal signal for the unbounded shard planner.
        # Preserve an exact candidate body/digest so the caller can prove why no
        # paid POST was made and partition before retrying.
        if isinstance(requested_max, int) and requested_max > 0:
            physical["max_completion_tokens"] = requested_max
        serialized = json.dumps(
            physical,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        context_length = context_envelope.get("context_length")
        token_upper = (
            len(serialized) + CRITIC_REQUEST_PROTOCOL_TOKEN_UPPER_BOUND
        )
        return {
            "request_payload": physical,
            "request_utf8_bytes": len(serialized),
            "request_sha256": hashlib.sha256(serialized).hexdigest(),
            "input_token_upper_bound": token_upper,
            "effective_max_completion_tokens": None,
            "context_headroom_tokens": (
                int(context_length) - token_upper
                if isinstance(context_length, int)
                and not isinstance(context_length, bool)
                else None
            ),
            "fits_model_envelope": False,
            "preflight_error": str(exc),
        }
    if isinstance(resolved_max, int) and resolved_max > 0:
        physical["max_completion_tokens"] = resolved_max
    serialized = json.dumps(
        physical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    estimate = dict(resolved_envelope.get("request_estimate") or {})
    if (
        estimate.get("serialized_request_utf8_bytes") != len(serialized)
        or estimate.get("request_sha256")
        != hashlib.sha256(serialized).hexdigest()
    ):
        raise OpenRouterError(
            "Critic physical-request preflight diverged from transport estimate"
        )
    return {
        "request_payload": physical,
        "request_utf8_bytes": len(serialized),
        "request_sha256": hashlib.sha256(serialized).hexdigest(),
        "input_token_upper_bound": estimate.get("input_token_upper_bound"),
        "effective_max_completion_tokens": resolved_max,
        "context_headroom_tokens": resolved_envelope.get(
            "context_headroom_tokens"
        ),
        "fits_model_envelope": True,
        "preflight_error": None,
    }


def _critic_messages_physical_request_utf8_bytes(
    *,
    messages: list[dict[str, Any]],
    schema_name: str,
    reasoning_effort: str,
    temperature: float,
    context_envelope: dict[str, Any],
) -> int:
    """Size the same canonical physical body sent by ``_critic_atomic_chat``."""

    physical: dict[str, Any] = {
        "model": CRITIC_MODEL,
        "messages": messages,
        "temperature": temperature,
        "reasoning": {
            "effort": reasoning_effort,
            "exclude": True,
        },
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": CRITIC_SCHEMA,
            },
        },
    }
    policy_fields, _policy = web_request_policy(
        model=CRITIC_MODEL,
        policy=WebSearchPolicy.FORBIDDEN,
    )
    physical.update(policy_fields)
    requested_max = context_envelope.get("max_completion_tokens")
    if not isinstance(requested_max, int) or isinstance(requested_max, bool):
        requested_max = context_envelope.get("reserved_output_tokens")
    if not isinstance(requested_max, int) or isinstance(requested_max, bool):
        requested_max = None
    try:
        resolved_max, _resolved = _apply_model_output_headroom(
            payload=physical,
            output_envelope=context_envelope,
            requested_max=requested_max,
        )
    except OpenRouterError as exc:
        if "no completion headroom" not in str(exc):
            raise
        if isinstance(requested_max, int) and requested_max > 0:
            physical["max_completion_tokens"] = requested_max
        return _json_utf8_bytes_canonical(physical)
    if isinstance(resolved_max, int) and resolved_max > 0:
        physical["max_completion_tokens"] = resolved_max
    return _json_utf8_bytes_canonical(physical)


def _json_utf8_bytes_canonical(value: Any) -> int:
    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )


async def _critic_input_budget_bytes() -> tuple[int, dict[str, Any]]:
    """Resolve a conservative input window from current model metadata.

    One UTF-8 byte is budgeted as one token.  This intentionally
    overestimates normal Russian/English JSON tokenization.  The model's
    advertised maximum output remains reserved, while no second arbitrary
    fixed margin is subtracted from the exactly serialized request budget.
    """

    envelope = await model_output_envelope(CRITIC_MODEL)
    context_length = envelope.get("context_length")
    maximum_output = envelope.get("max_completion_tokens")
    if not (
        isinstance(context_length, int)
        and not isinstance(context_length, bool)
        and context_length > 0
        and isinstance(maximum_output, int)
        and not isinstance(maximum_output, bool)
        and maximum_output > 0
    ):
        raise OpenRouterError(
            "Critic model envelope must prove positive context and output maxima"
        )
    available = (
        context_length
        - maximum_output
        - CRITIC_REQUEST_PROTOCOL_TOKEN_UPPER_BOUND
    )
    if available <= 0:
        raise OpenRouterError(
            "Critic model metadata leaves no safe input window: "
            f"context={context_length}, max_output={maximum_output}"
        )
    return available, {
        **_json_safe(envelope),
        "input_accounting": "one_utf8_byte_per_token_upper_bound",
        "reserved_output_tokens": maximum_output,
        "fixed_context_safety_tokens": (
            CRITIC_REQUEST_PROTOCOL_TOKEN_UPPER_BOUND
        ),
        "input_budget_bytes": available,
    }


def _validated_review_schema(
    review: dict[str, Any],
    *,
    owner: str,
) -> dict[str, Any]:
    errors = sorted(
        Draft202012Validator(CRITIC_SCHEMA).iter_errors(review),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        details = "; ".join(error.message for error in errors)
        raise OpenRouterError(
            f"{owner} violated the critic JSON schema: {details}"
        )
    return _normalize_review(review)


def _ordered_complete_answers(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate the raw corpus and return a deterministic answer order."""

    raw_answers = payload.get("answers") or []
    if not isinstance(raw_answers, list):
        raise OpenRouterError("Critic audit payload answers must be a list")
    by_id: dict[int, dict[str, Any]] = {}
    for position, raw_answer in enumerate(raw_answers):
        if not isinstance(raw_answer, dict):
            raise OpenRouterError(
                f"Critic answer at position {position} is not an object"
            )
        answer_id = raw_answer.get("answer_id")
        if (
            not isinstance(answer_id, int)
            or isinstance(answer_id, bool)
        ):
            raise OpenRouterError(
                f"Critic answer at position {position} has no integer answer_id"
            )
        if answer_id in by_id:
            raise OpenRouterError(
                f"Critic raw corpus contains duplicate answer_id={answer_id}"
            )
        answer = dict(raw_answer)
        raw_text = str(answer.get("raw_answer") or "")
        raw_chars = answer.get("raw_answer_char_count")
        if (
            isinstance(raw_chars, int)
            and not isinstance(raw_chars, bool)
            and raw_chars != len(raw_text)
        ):
            raise OpenRouterError(
                "Critic raw answer character count mismatch for "
                f"answer_id={answer_id}"
            )
        if answer.get("raw_answer_included") is False and (
            raw_text or (isinstance(raw_chars, int) and raw_chars > 0)
        ):
            raise OpenRouterError(
                "Critic raw corpus contains manifest-only nonempty evidence for "
                f"answer_id={answer_id}"
            )
        digest = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
        saved_digest = answer.get("raw_answer_sha256")
        if saved_digest is not None and str(saved_digest) != digest:
            raise OpenRouterError(
                f"Critic raw answer digest mismatch for answer_id={answer_id}"
            )
        manifest = answer.get("raw_answer_manifest")
        if isinstance(manifest, dict):
            manifest_digest = manifest.get("source_sha256")
            manifest_chars = manifest.get("source_chars")
            if manifest_digest is not None and str(manifest_digest) != digest:
                raise OpenRouterError(
                    "Critic raw answer manifest digest mismatch for "
                    f"answer_id={answer_id}"
                )
            if (
                isinstance(manifest_chars, int)
                and not isinstance(manifest_chars, bool)
                and manifest_chars != len(raw_text)
            ):
                raise OpenRouterError(
                    "Critic raw answer manifest length mismatch for "
                    f"answer_id={answer_id}"
                )
        by_id[answer_id] = answer
    return [by_id[answer_id] for answer_id in sorted(by_id)]


def _critic_complete_answer_manifest(
    answer: dict[str, Any],
) -> dict[str, Any]:
    raw_manifest = answer.get("raw_answer_manifest")
    raw_manifest = raw_manifest if isinstance(raw_manifest, dict) else {}
    return {
        **_critic_answer_index(answer),
        "raw_answer_manifest": {
            "version": raw_manifest.get("version"),
            "document_id": raw_manifest.get("document_id"),
            "source_sha256": raw_manifest.get("source_sha256"),
            "source_chars": raw_manifest.get("source_chars"),
            "source_utf8_bytes": raw_manifest.get("source_utf8_bytes"),
            "unit_count": raw_manifest.get("unit_count"),
            "manifest_sha256": (
                _stable_sha256(raw_manifest) if raw_manifest else None
            ),
        },
    }


def _critic_leaf_payload(
    payload: dict[str, Any],
    *,
    answers: list[dict[str, Any]],
    complete_answers: list[dict[str, Any]],
    leaf_index: int,
    leaf_count: int,
    audit_payload_sha256: str | None = None,
    shared_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    shared = (
        dict(shared_context)
        if shared_context is not None
        else {
            key: value
            for key, value in payload.items()
            if key != "answers"
        }
    )
    complete_ids = [int(answer["answer_id"]) for answer in complete_answers]
    assigned_ids = [int(answer["answer_id"]) for answer in answers]
    complete_index = [
        _critic_answer_index(answer) for answer in complete_answers
    ]
    return {
        **shared,
        "critic_map_partition": {
            "version": CRITIC_MAP_REDUCE_VERSION,
            "audit_payload_sha256": (
                audit_payload_sha256 or _stable_sha256(payload)
            ),
            "leaf_index": leaf_index,
            "leaf_count": leaf_count,
            "complete_answer_count": len(complete_ids),
            "complete_answer_index_sha256": _stable_sha256(complete_index),
            "assigned_answer_ids": assigned_ids,
            "assigned_answer_count": len(assigned_ids),
            "each_answer_assigned_exactly_once": True,
        },
        "complete_answer_index_manifest": {
            "answer_count": len(complete_ids),
            "index_sha256": _stable_sha256(complete_index),
            "ids_sha256": _stable_sha256(complete_ids),
        },
        "assigned_answer_index": [
            _critic_answer_index(answer) for answer in answers
        ],
        "complete_answer_manifests": [
            _critic_complete_answer_manifest(answer)
            for answer in answers
        ],
        "answers": answers,
    }


def _shared_context_source(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key != "answers"
    }


def _shared_context_manifest_brief(
    manifest: dict[str, Any],
) -> dict[str, Any]:
    return {
        "version": manifest.get("version"),
        "source_sha256": manifest.get("source_sha256"),
        "source_chars": manifest.get("source_chars"),
        "source_utf8_bytes": manifest.get("source_utf8_bytes"),
        "core_unit_count": manifest.get("core_unit_count"),
        "core_unit_ids_sha256": _stable_sha256(
            manifest.get("core_unit_ids") or []
        ),
        "lossless_manifest_sha256": manifest.get(
            "lossless_manifest_sha256"
        ),
        "exact_core_accounting": bool(
            manifest.get("exact_core_accounting")
        ),
    }


def _shared_context_pending_digest(
    manifest: dict[str, Any],
    *,
    input_budget_bytes: int,
) -> dict[str, Any]:
    # This is reusable space inside one physical answer/context join, not a
    # semantic corpus ceiling.  The source may create any number of calls.
    reserve = max(4_096, input_budget_bytes // 3)
    return {
        "shared_context_manifest": _shared_context_manifest_brief(manifest),
        "shared_context_digest": {
            "status": "pending_lossless_answer_context_join",
            "reserved_utf8_bytes": reserve,
            # The placeholder makes the base answer planner leave room for one
            # exact fact shard. It is removed before every provider call.
            "reserved_content": " " * reserve,
        },
    }


def _shared_context_leaf_payload(
    unit: TextUnit,
    *,
    source_manifest: dict[str, Any],
    unit_count: int,
) -> dict[str, Any]:
    return {
        "critic_shared_context_partition": {
            "version": CRITIC_MAP_REDUCE_VERSION,
            "mode": "shared_context_fragment",
            "source_sha256": source_manifest["source_sha256"],
            "source_chars": source_manifest["source_chars"],
            "source_utf8_bytes": source_manifest["source_utf8_bytes"],
            "lossless_manifest_sha256": _stable_sha256(source_manifest),
            "unit_id": unit.unit_id,
            "unit_index": unit.index,
            "unit_count": unit_count,
            "core_start_char": unit.start_char,
            "core_end_char": unit.end_char,
            "core_sha256": unit.sha256,
            "context_start_char": unit.context_start_char,
            "context_end_char": unit.context_end_char,
            "context_sha256": unit.context_sha256,
            "core_start_in_context": unit.core_start_in_context,
            "core_end_in_context": unit.core_end_in_context,
            "ownership_rule": "first_material_character_in_core",
            "overlap_counts_toward_core_coverage": False,
        },
        "shared_context_json_fragment": unit.context_text,
        "answers": [],
    }


def _build_shared_context_tasks(
    payload: dict[str, Any],
    *,
    iteration: int,
    max_iterations: int,
    recovery_final: bool,
    input_budget_bytes: int,
    context_envelope: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Partition canonical shared JSON with exact lossless core accounting."""

    source = json.dumps(
        _shared_context_source(payload),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    )
    target_chars = max(256, min(len(source), input_budget_bytes // 2))
    while True:
        units, text_manifest = split_lossless_text(
            source,
            document_id=(
                "critic-shared-context-" + _stable_sha256(payload)[:20]
            ),
            target_chars=target_chars,
            context_overlap_chars=DEFAULT_CONTEXT_OVERLAP_CHARS,
        )
        source_manifest = text_manifest.as_dict()
        if _verify_fragment_core_accounting(units, source_manifest) != source:
            raise OpenRouterError(
                "Critic shared-context reconstruction changed canonical JSON"
            )
        tasks: list[dict[str, Any]] = []
        request_sizes: list[int] = []
        for unit in units:
            leaf_payload = _shared_context_leaf_payload(
                unit,
                source_manifest=source_manifest,
                unit_count=len(units),
            )
            request_size = _critic_request_utf8_bytes(
                leaf_payload,
                iteration=iteration,
                max_iterations=max_iterations,
                recovery_final=recovery_final,
                shared_context_leaf=True,
                schema_name=(
                    f"aiv_analysis_critic_{iteration}_shared_context_"
                    f"{unit.index}"
                ),
                context_envelope=context_envelope,
            )
            tasks.append(
                {
                    "kind": "shared_context",
                    "unit_id": unit.unit_id,
                    "unit_index": unit.index,
                    "payload": leaf_payload,
                }
            )
            request_sizes.append(request_size)
        largest = max(request_sizes, default=0)
        if largest <= input_budget_bytes:
            core_ids = [unit.unit_id for unit in units]
            core_chars = sum(unit.end_char - unit.start_char for unit in units)
            context_chars = sum(len(unit.context_text) for unit in units)
            return tasks, {
                "version": CRITIC_MAP_REDUCE_VERSION,
                "mode": "lossless_shared_context",
                "source_sha256": source_manifest["source_sha256"],
                "source_chars": source_manifest["source_chars"],
                "source_utf8_bytes": source_manifest["source_utf8_bytes"],
                "target_core_window_chars": target_chars,
                "context_overlap_chars": DEFAULT_CONTEXT_OVERLAP_CHARS,
                "core_unit_count": len(units),
                "core_unit_ids": core_ids,
                "submitted_core_chars": core_chars,
                "submitted_context_chars": context_chars,
                "overlap_chars_excluded_from_coverage": (
                    context_chars - core_chars
                ),
                "shared_context_request_utf8_bytes": request_sizes,
                "lossless_manifest_sha256": _stable_sha256(source_manifest),
                "lossless_manifest": source_manifest,
                "missing_core_unit_ids": [],
                "duplicate_core_unit_ids": [],
                "exact_core_accounting": True,
            }
        if target_chars == 256:
            raise OpenRouterError(
                "Even the minimum shared-context fragment cannot fit the "
                "critic model context without truncation: "
                f"request_bytes={largest}, budget_bytes={input_budget_bytes}"
            )
        ratio = max(0.1, min(0.9, input_budget_bytes / max(1, largest)))
        target_chars = max(
            256,
            min(target_chars - 1, int(target_chars * ratio)),
        )


def _build_critic_leaf_payloads(
    payload: dict[str, Any],
    *,
    iteration: int,
    max_iterations: int,
    recovery_final: bool,
    input_budget_bytes: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Greedily pack whole answers; no raw answer can cross a leaf boundary."""

    complete_answers = _ordered_complete_answers(payload)
    audit_payload_sha256 = _stable_sha256(payload)
    if not complete_answers:
        raise OpenRouterError(
            "Oversized critic payload has no answer corpus to partition"
        )
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for answer in complete_answers:
        candidate = [*current, answer]
        provisional = _critic_leaf_payload(
            payload,
            answers=candidate,
            complete_answers=complete_answers,
            leaf_index=999_999,
            leaf_count=999_999,
            audit_payload_sha256=audit_payload_sha256,
        )
        request_bytes = _critic_request_utf8_bytes(
            provisional,
            iteration=iteration,
            max_iterations=max_iterations,
            recovery_final=recovery_final,
            map_leaf=True,
        )
        if request_bytes <= input_budget_bytes:
            current = candidate
            continue
        if not current:
            raise OpenRouterError(
                "One complete raw answer cannot fit the critic model context; "
                "the harness refuses to truncate or split it: "
                f"answer_id={answer['answer_id']}, request_bytes={request_bytes}, "
                f"budget_bytes={input_budget_bytes}"
            )
        groups.append(current)
        current = [answer]
        single = _critic_leaf_payload(
            payload,
            answers=current,
            complete_answers=complete_answers,
            leaf_index=999_999,
            leaf_count=999_999,
            audit_payload_sha256=audit_payload_sha256,
        )
        single_bytes = _critic_request_utf8_bytes(
            single,
            iteration=iteration,
            max_iterations=max_iterations,
            recovery_final=recovery_final,
            map_leaf=True,
        )
        if single_bytes > input_budget_bytes:
            raise OpenRouterError(
                "One complete raw answer cannot fit the critic model context; "
                "the harness refuses to truncate or split it: "
                f"answer_id={answer['answer_id']}, request_bytes={single_bytes}, "
                f"budget_bytes={input_budget_bytes}"
            )
    if current:
        groups.append(current)

    leaf_count = len(groups)
    leaves = [
        _critic_leaf_payload(
            payload,
            answers=group,
            complete_answers=complete_answers,
            leaf_index=index,
            leaf_count=leaf_count,
            audit_payload_sha256=audit_payload_sha256,
        )
        for index, group in enumerate(groups)
    ]
    assigned_ids = [
        int(answer["answer_id"])
        for group in groups
        for answer in group
    ]
    complete_ids = [int(answer["answer_id"]) for answer in complete_answers]
    if assigned_ids != complete_ids or len(assigned_ids) != len(set(assigned_ids)):
        raise OpenRouterError(
            "Critic map partition failed exact answer accounting"
        )
    request_sizes = [
        _critic_request_utf8_bytes(
            leaf,
            iteration=iteration,
            max_iterations=max_iterations,
            recovery_final=recovery_final,
            map_leaf=True,
        )
        for leaf in leaves
    ]
    if any(size > input_budget_bytes for size in request_sizes):
        raise OpenRouterError(
            "Finalized critic leaf exceeded the resolved input window"
        )
    return leaves, {
        "version": CRITIC_MAP_REDUCE_VERSION,
        "audit_payload_sha256": audit_payload_sha256,
        "complete_answer_count": len(complete_ids),
        "complete_answer_ids": complete_ids,
        "leaf_count": leaf_count,
        "leaf_answer_ids": [
            leaf["critic_map_partition"]["assigned_answer_ids"]
            for leaf in leaves
        ],
        "leaf_request_utf8_bytes": request_sizes,
        "input_budget_bytes": input_budget_bytes,
        "missing_answer_ids": [],
        "duplicate_answer_ids": [],
        "exact_accounting": True,
    }


def _verify_fragment_core_accounting(
    units: list[TextUnit],
    manifest: dict[str, Any],
) -> str:
    """Fail closed unless every non-overlapping core appears exactly once."""

    unit_ids = [unit.unit_id for unit in units]
    expected_units = manifest.get("units")
    expected_ids = [
        str(item.get("unit_id") or "")
        for item in expected_units or []
        if isinstance(item, dict)
    ]
    if len(unit_ids) != len(set(unit_ids)):
        raise OpenRouterError(
            "Critic fragment plan contains duplicate core unit ids"
        )
    if unit_ids != expected_ids:
        missing = sorted(set(expected_ids) - set(unit_ids))
        unexpected = sorted(set(unit_ids) - set(expected_ids))
        raise OpenRouterError(
            "Critic fragment plan failed core unit accounting: "
            f"missing={missing}, unexpected={unexpected}"
        )
    try:
        reconstructed = verify_units(units, manifest)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise OpenRouterError(
            f"Critic fragment reconstruction failed: {exc}"
        ) from exc
    submitted_core_chars = sum(
        unit.end_char - unit.start_char for unit in units
    )
    if submitted_core_chars != int(manifest.get("source_chars") or 0):
        raise OpenRouterError(
            "Critic fragment core ranges do not cover the complete answer"
        )
    return reconstructed


def _critic_fragment_answer(
    answer: dict[str, Any],
    unit: TextUnit,
    *,
    source_manifest: dict[str, Any],
) -> dict[str, Any]:
    context = unit.context_text
    context_manifest = {
        "version": CRITIC_MAP_REDUCE_VERSION,
        "document_id": f"{unit.unit_id}:context",
        "source_sha256": unit.context_sha256,
        "source_chars": len(context),
        "source_utf8_bytes": len(context.encode("utf-8")),
        "unit_count": 1,
        "units": [
            {
                "unit_id": unit.unit_id,
                "core_start_char": unit.start_char,
                "core_end_char": unit.end_char,
                "core_sha256": unit.sha256,
                "context_start_char": unit.context_start_char,
                "context_end_char": unit.context_end_char,
                "context_sha256": unit.context_sha256,
                "core_start_in_context": unit.core_start_in_context,
                "core_end_in_context": unit.core_end_in_context,
            }
        ],
    }
    return {
        **answer,
        "raw_answer": context,
        "raw_answer_sha256": unit.context_sha256,
        "raw_answer_char_count": len(context),
        "raw_answer_manifest": context_manifest,
        "raw_answer_included": True,
        "raw_answer_truncated": False,
        "raw_answer_omission_reason": None,
        "critic_fragment": {
            "version": CRITIC_MAP_REDUCE_VERSION,
            "answer_id": int(answer["answer_id"]),
            "source_sha256": source_manifest["source_sha256"],
            "source_chars": source_manifest["source_chars"],
            "unit_id": unit.unit_id,
            "unit_index": unit.index,
            "unit_count": source_manifest["unit_count"],
            "core_start_char": unit.start_char,
            "core_end_char": unit.end_char,
            "core_sha256": unit.sha256,
            "context_start_char": unit.context_start_char,
            "context_end_char": unit.context_end_char,
            "context_sha256": unit.context_sha256,
            "core_start_in_context": unit.core_start_in_context,
            "core_end_in_context": unit.core_end_in_context,
            "ownership_rule": "first_decisive_evidence_character_in_core",
            "overlap_counts_toward_core_coverage": False,
        },
    }


def _fragment_answer_payloads(
    payload: dict[str, Any],
    *,
    answer: dict[str, Any],
    complete_answers: list[dict[str, Any]],
    iteration: int,
    max_iterations: int,
    recovery_final: bool,
    input_budget_bytes: int,
    audit_payload_sha256: str,
    shared_context: dict[str, Any] | None = None,
    context_envelope: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Find a physical fragment window without ever limiting source length."""

    raw_text = str(answer.get("raw_answer") or "")
    target_chars = max(256, min(len(raw_text), input_budget_bytes // 2))
    while True:
        units, text_manifest = split_lossless_text(
            raw_text,
            document_id=f"critic-answer-{int(answer['answer_id'])}",
            target_chars=target_chars,
            context_overlap_chars=DEFAULT_CONTEXT_OVERLAP_CHARS,
        )
        manifest = text_manifest.as_dict()
        if _verify_fragment_core_accounting(units, manifest) != raw_text:
            raise OpenRouterError(
                "Critic fragment reconstruction changed the raw answer"
            )
        fragment_payloads: list[dict[str, Any]] = []
        request_sizes: list[int] = []
        for unit in units:
            fragment_answer = _critic_fragment_answer(
                answer,
                unit,
                source_manifest=manifest,
            )
            fragment_payload = _critic_leaf_payload(
                payload,
                answers=[fragment_answer],
                complete_answers=complete_answers,
                leaf_index=unit.index,
                leaf_count=manifest["unit_count"],
                audit_payload_sha256=audit_payload_sha256,
                shared_context=shared_context,
            )
            fragment_payload["critic_map_partition"].update(
                {
                    "mode": "answer_fragment",
                    "source_answer_id": int(answer["answer_id"]),
                    "assigned_core_unit_ids": [unit.unit_id],
                    "assigned_core_unit_count": 1,
                }
            )
            request_size = _critic_request_utf8_bytes(
                fragment_payload,
                iteration=iteration,
                max_iterations=max_iterations,
                recovery_final=recovery_final,
                map_leaf=True,
                fragment_leaf=True,
                schema_name=(
                    f"aiv_analysis_critic_{iteration}_answer_"
                    f"{answer['answer_id']}_fragment_{unit.index}"
                ),
                context_envelope=context_envelope,
            )
            fragment_payloads.append(fragment_payload)
            request_sizes.append(request_size)
        largest = max(request_sizes, default=0)
        if largest <= input_budget_bytes:
            core_ids = [unit.unit_id for unit in units]
            core_chars = sum(
                unit.end_char - unit.start_char for unit in units
            )
            context_chars = sum(len(unit.context_text) for unit in units)
            return fragment_payloads, {
                "version": CRITIC_MAP_REDUCE_VERSION,
                "answer_id": int(answer["answer_id"]),
                "source_sha256": manifest["source_sha256"],
                "source_chars": manifest["source_chars"],
                "source_utf8_bytes": manifest["source_utf8_bytes"],
                "target_core_window_chars": target_chars,
                "context_overlap_chars": DEFAULT_CONTEXT_OVERLAP_CHARS,
                "core_unit_count": len(units),
                "core_unit_ids": core_ids,
                "submitted_core_chars": core_chars,
                "submitted_context_chars": context_chars,
                "overlap_chars_excluded_from_coverage": (
                    context_chars - core_chars
                ),
                "fragment_request_utf8_bytes": request_sizes,
                "lossless_manifest": manifest,
                "missing_core_unit_ids": [],
                "duplicate_core_unit_ids": [],
                "exact_core_accounting": True,
            }
        if target_chars == 256:
            raise OpenRouterError(
                "Even the minimum semantic-overlap critic fragment cannot "
                "fit the model context without truncation: "
                f"answer_id={answer['answer_id']}, request_bytes={largest}, "
                f"budget_bytes={input_budget_bytes}"
            )
        # Shrink only the per-request core window.  The loop may create any
        # number of units, so this never becomes a corpus or answer cap.
        ratio = max(0.1, min(0.9, input_budget_bytes / max(1, largest)))
        next_target = max(256, min(target_chars - 1, int(target_chars * ratio)))
        target_chars = next_target


def _build_critic_map_plan(
    payload: dict[str, Any],
    *,
    iteration: int,
    max_iterations: int,
    recovery_final: bool,
    input_budget_bytes: int,
    context_envelope: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Plan whole-answer leaves plus lossless fragments for oversized answers."""

    complete_answers = _ordered_complete_answers(payload)
    audit_payload_sha256 = _stable_sha256(payload)
    if not complete_answers:
        raise OpenRouterError(
            "Oversized critic payload has no answer corpus to partition"
        )
    shared_context_tasks: list[dict[str, Any]] = []
    shared_context_manifest: dict[str, Any] | None = None
    answer_shared_context: dict[str, Any] | None = None
    shared_probe = _critic_leaf_payload(
        payload,
        answers=[],
        complete_answers=complete_answers,
        leaf_index=999_999,
        leaf_count=999_999,
        audit_payload_sha256=audit_payload_sha256,
    )
    shared_probe_bytes = _critic_request_utf8_bytes(
        shared_probe,
        iteration=iteration,
        max_iterations=max_iterations,
        recovery_final=recovery_final,
        map_leaf=True,
        schema_name=f"aiv_analysis_critic_{iteration}_leaf_999999999",
        context_envelope=context_envelope,
    )
    if shared_probe_bytes > input_budget_bytes:
        shared_context_tasks, shared_context_manifest = (
            _build_shared_context_tasks(
                payload,
                iteration=iteration,
                max_iterations=max_iterations,
                recovery_final=recovery_final,
                input_budget_bytes=input_budget_bytes,
                context_envelope=context_envelope,
            )
        )
        answer_shared_context = _shared_context_pending_digest(
            shared_context_manifest,
            input_budget_bytes=input_budget_bytes,
        )
    whole_answers: list[dict[str, Any]] = []
    fragmented: list[tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]] = []
    for answer in complete_answers:
        single = _critic_leaf_payload(
            payload,
            answers=[answer],
            complete_answers=complete_answers,
            leaf_index=999_999,
            leaf_count=999_999,
            audit_payload_sha256=audit_payload_sha256,
            shared_context=answer_shared_context,
        )
        single_bytes = _critic_request_utf8_bytes(
            single,
            iteration=iteration,
            max_iterations=max_iterations,
            recovery_final=recovery_final,
            map_leaf=True,
            schema_name=f"aiv_analysis_critic_{iteration}_leaf_999999999",
            context_envelope=context_envelope,
        )
        if single_bytes <= input_budget_bytes:
            whole_answers.append(answer)
            continue
        fragments, fragment_manifest = _fragment_answer_payloads(
            payload,
            answer=answer,
            complete_answers=complete_answers,
            iteration=iteration,
            max_iterations=max_iterations,
            recovery_final=recovery_final,
            input_budget_bytes=input_budget_bytes,
            audit_payload_sha256=audit_payload_sha256,
            shared_context=answer_shared_context,
            context_envelope=context_envelope,
        )
        fragmented.append((answer, fragments, fragment_manifest))

    whole_groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for answer in whole_answers:
        candidate = [*current, answer]
        provisional = _critic_leaf_payload(
            payload,
            answers=candidate,
            complete_answers=complete_answers,
            leaf_index=999_999,
            leaf_count=999_999,
            audit_payload_sha256=audit_payload_sha256,
            shared_context=answer_shared_context,
        )
        size = _critic_request_utf8_bytes(
            provisional,
            iteration=iteration,
            max_iterations=max_iterations,
            recovery_final=recovery_final,
            map_leaf=True,
            schema_name=f"aiv_analysis_critic_{iteration}_leaf_999999999",
            context_envelope=context_envelope,
        )
        if size <= input_budget_bytes:
            current = candidate
        else:
            if not current:
                raise OpenRouterError(
                    "Critic whole-answer packing failed after fit preflight"
                )
            whole_groups.append(current)
            current = [answer]
    if current:
        whole_groups.append(current)

    fragment_leaf_count = sum(
        len(fragment_payloads)
        for _answer, fragment_payloads, _manifest in fragmented
    )
    shared_context_leaf_count = len(shared_context_tasks)
    total_leaf_count = (
        len(whole_groups)
        + fragment_leaf_count
        + shared_context_leaf_count
    )
    whole_leaves = [
        _critic_leaf_payload(
            payload,
            answers=group,
            complete_answers=complete_answers,
            leaf_index=index,
            leaf_count=total_leaf_count,
            audit_payload_sha256=audit_payload_sha256,
            shared_context=answer_shared_context,
        )
        for index, group in enumerate(whole_groups)
    ]
    whole_sizes = [
        _critic_request_utf8_bytes(
            leaf,
            iteration=iteration,
            max_iterations=max_iterations,
            recovery_final=recovery_final,
            map_leaf=True,
            schema_name=f"aiv_analysis_critic_{iteration}_leaf_{index}",
            context_envelope=context_envelope,
        )
        for index, leaf in enumerate(whole_leaves)
    ]
    fragment_tasks: list[dict[str, Any]] = list(shared_context_tasks)
    fragment_manifests: list[dict[str, Any]] = []
    for answer, fragment_payloads, fragment_manifest in fragmented:
        fragment_manifests.append(fragment_manifest)
        for fragment_payload in fragment_payloads:
            fragment = fragment_payload["answers"][0]["critic_fragment"]
            fragment_tasks.append(
                {
                    "answer_id": int(answer["answer_id"]),
                    "unit_id": fragment["unit_id"],
                    "unit_index": fragment["unit_index"],
                    "payload": fragment_payload,
                }
            )

    whole_ids = [
        int(answer["answer_id"])
        for group in whole_groups
        for answer in group
    ]
    fragmented_ids = [int(answer["answer_id"]) for answer, _p, _m in fragmented]
    covered_ids = sorted([*whole_ids, *fragmented_ids])
    complete_ids = [int(answer["answer_id"]) for answer in complete_answers]
    if covered_ids != complete_ids or len(covered_ids) != len(set(covered_ids)):
        raise OpenRouterError(
            "Critic map plan failed exact answer-level accounting"
        )
    fragment_sizes = [
        size
        for manifest in fragment_manifests
        for size in manifest["fragment_request_utf8_bytes"]
    ]
    shared_context_sizes = (
        list(
            shared_context_manifest.get(
                "shared_context_request_utf8_bytes",
                [],
            )
        )
        if shared_context_manifest is not None
        else []
    )
    return whole_leaves, fragment_tasks, {
        "version": CRITIC_MAP_REDUCE_VERSION,
        "audit_payload_sha256": audit_payload_sha256,
        "complete_answer_count": len(complete_ids),
        "complete_answer_ids": complete_ids,
        "whole_answer_ids": whole_ids,
        "fragmented_answer_ids": fragmented_ids,
        "leaf_count": total_leaf_count,
        "whole_leaf_count": len(whole_leaves),
        "fragment_leaf_count": fragment_leaf_count,
        "shared_context_leaf_count": shared_context_leaf_count,
        "shared_context_partitioned": shared_context_manifest is not None,
        "shared_context": (
            {
                **_shared_context_manifest_brief(shared_context_manifest),
                "shared_probe_utf8_bytes": shared_probe_bytes,
                "digest_reserve_bytes": (
                    answer_shared_context["shared_context_digest"][
                        "reserved_utf8_bytes"
                    ]
                    if answer_shared_context is not None
                    else 0
                ),
            }
            if shared_context_manifest is not None
            else None
        ),
        "leaf_answer_ids": [
            leaf["critic_map_partition"]["assigned_answer_ids"]
            for leaf in whole_leaves
        ],
        "whole_leaf_answer_ids": [
            leaf["critic_map_partition"]["assigned_answer_ids"]
            for leaf in whole_leaves
        ],
        "leaf_request_utf8_bytes": [
            *whole_sizes,
            *fragment_sizes,
            *shared_context_sizes,
        ],
        "fragmented_answers": fragment_manifests,
        "input_budget_bytes": input_budget_bytes,
        "missing_answer_ids": [],
        "duplicate_answer_ids": [],
        "exact_accounting": True,
    }


def _stable_unique_dicts(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in values:
        digest = _stable_sha256(value)
        if digest in seen:
            continue
        seen.add(digest)
        output.append(dict(value))
    return output


def _stable_unique_strings(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for raw_value in values:
        value = str(raw_value).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _critic_review_pass_seed(summary: str) -> dict[str, Any]:
    return {
        "verdict": "pass",
        "summary": summary,
        "anomalies": [],
        "policy_adjustments": [],
        "annotation_guidance": "",
        "acceptance_checks": [],
    }


def _merge_reviews_preserving_material_findings(
    reviews: list[dict[str, Any]],
    proposed: dict[str, Any],
) -> dict[str, Any]:
    """Apply a deterministic verdict floor and preserve every material leaf."""

    rank = {"pass": 0, "revise": 1, "block": 2}
    all_reviews = [*reviews, proposed]
    verdict = max(
        (str(review.get("verdict") or "block") for review in all_reviews),
        key=lambda value: rank.get(value, 2),
    )
    source_anomalies = [
        dict(anomaly)
        for review in reviews
        for anomaly in review.get("anomalies") or []
        if isinstance(anomaly, dict)
    ]
    proposed_anomalies = [
        dict(anomaly)
        for anomaly in proposed.get("anomalies") or []
        if isinstance(anomaly, dict)
    ]
    if verdict == "pass":
        # One observation per warning code keeps the existing deterministic
        # gate contract while unioning every referenced answer/entity.
        observations: dict[str, dict[str, Any]] = {}
        for anomaly in [*source_anomalies, *proposed_anomalies]:
            code = str(anomaly.get("code") or "other")
            existing = observations.get(code)
            if existing is None:
                observations[code] = dict(anomaly)
                continue
            if len(str(anomaly.get("finding") or "")) > len(
                str(existing.get("finding") or "")
            ):
                existing["finding"] = anomaly.get("finding")
            existing["answer_ids"] = sorted(
                {
                    value
                    for value in [
                        *(existing.get("answer_ids") or []),
                        *(anomaly.get("answer_ids") or []),
                    ]
                    if isinstance(value, int) and not isinstance(value, bool)
                }
            )
            existing["entities"] = _stable_unique_strings(
                [
                    *(existing.get("entities") or []),
                    *(anomaly.get("entities") or []),
                ]
            )
        anomalies = list(observations.values())
    else:
        # Leaf findings precede reducer additions so deterministic order is
        # independent of provider wording or response timing.
        anomalies = _stable_unique_dicts(
            [*source_anomalies, *proposed_anomalies]
        )
    adjustments = _stable_unique_dicts(
        [
            dict(adjustment)
            for review in all_reviews
            for adjustment in review.get("policy_adjustments") or []
            if isinstance(adjustment, dict)
        ]
    )
    guidance_parts = _stable_unique_strings(
        [
            str(review.get("annotation_guidance") or "")
            for review in all_reviews
        ]
    )
    checks = _stable_unique_strings(
        [
            str(check)
            for review in all_reviews
            for check in review.get("acceptance_checks") or []
        ]
    )
    merged = {
        "verdict": verdict,
        "summary": str(proposed.get("summary") or "").strip()
        or (
            "Итоговый critic-verdict собран из всех частей raw-корпуса без "
            "пропусков."
        ),
        "anomalies": anomalies,
        "policy_adjustments": [] if verdict == "pass" else adjustments,
        "annotation_guidance": (
            "" if verdict == "pass" else "\n\n".join(guidance_parts)
        ),
        "acceptance_checks": checks,
    }
    if verdict == "pass" and not merged["acceptance_checks"]:
        merged["acceptance_checks"] = [
            "Все части raw-корпуса учтены, а их verdict сведены без "
            "понижения строгости."
        ]
    return _validated_review_schema(merged, owner="Critic map/reduce merger")


def _collapse_answer_overlap_duplicates(
    review: dict[str, Any],
) -> dict[str, Any]:
    """Collapse duplicate votes caused by semantic overlap, preserving text."""

    anomaly_groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    finding_parts: dict[tuple[Any, ...], list[str]] = {}
    for anomaly in review.get("anomalies") or []:
        if not isinstance(anomaly, dict):
            continue
        key = (
            str(anomaly.get("code") or "other"),
            str(anomaly.get("severity") or "observation"),
            tuple(sorted(
                value
                for value in anomaly.get("answer_ids") or []
                if isinstance(value, int) and not isinstance(value, bool)
            )),
            tuple(sorted(
                str(value).casefold()
                for value in anomaly.get("entities") or []
                if isinstance(value, str) and value.strip()
            )),
        )
        if key not in anomaly_groups:
            anomaly_groups[key] = dict(anomaly)
            finding_parts[key] = []
        finding = str(anomaly.get("finding") or "").strip()
        if finding and finding not in finding_parts[key]:
            finding_parts[key].append(finding)
        anomaly_groups[key]["answer_ids"] = list(key[2])
        anomaly_groups[key]["entities"] = _stable_unique_strings(
            [
                *(anomaly_groups[key].get("entities") or []),
                *(anomaly.get("entities") or []),
            ]
        )
    anomalies: list[dict[str, Any]] = []
    for key, anomaly in anomaly_groups.items():
        anomaly["finding"] = " ".join(finding_parts[key])
        anomalies.append(anomaly)

    adjustment_groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    reason_parts: dict[tuple[Any, ...], list[str]] = {}
    for adjustment in review.get("policy_adjustments") or []:
        if not isinstance(adjustment, dict):
            continue
        key = (
            str(adjustment.get("action") or ""),
            str(adjustment.get("entity_name") or "").casefold(),
            (
                str(adjustment.get("alias") or "").casefold()
                if adjustment.get("alias") is not None
                else None
            ),
            tuple(sorted(
                value
                for value in adjustment.get("answer_ids") or []
                if isinstance(value, int) and not isinstance(value, bool)
            )),
        )
        if key not in adjustment_groups:
            adjustment_groups[key] = dict(adjustment)
            reason_parts[key] = []
        reason = str(adjustment.get("reason") or "").strip()
        if reason and reason not in reason_parts[key]:
            reason_parts[key].append(reason)
        adjustment_groups[key]["answer_ids"] = list(key[3])
    adjustments: list[dict[str, Any]] = []
    for key, adjustment in adjustment_groups.items():
        adjustment["reason"] = " ".join(reason_parts[key])
        adjustments.append(adjustment)

    collapsed = {
        **review,
        "anomalies": anomalies,
        "policy_adjustments": adjustments,
    }
    return _validated_review_schema(
        collapsed,
        owner="Critic per-answer overlap deduplicator",
    )


async def _review_analysis_once(
    payload: dict[str, Any],
    *,
    iteration: int,
    max_iterations: int,
    recovery_final: bool,
    schema_name: str,
    map_leaf: bool = False,
    fragment_leaf: bool = False,
    shared_context_leaf: bool = False,
    context_join_leaf: bool = False,
    validate_schema: bool = False,
    audit_sink: CriticCallAuditSink | None = None,
    transport_audit_checkpoint: AuditCheckpoint | None = None,
    transport_resume_lookup: TransportResumeLookup | None = None,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Run one atomic critic verdict, with one bounded repair if allowed.

    Input evidence may already be a code-owned map/reduce shard. A physically
    limited verdict is never treated as a continuation prefix; it is replanned
    into bounded anomaly/policy/conclusion decisions. Authority is granted only
    when every independent decision shard completes, otherwise it fails closed.
    """

    messages = [
        {
            "role": "system",
            "content": _critic_system_prompt(
                recovery_final=recovery_final,
                map_leaf=map_leaf,
                fragment_leaf=fragment_leaf,
                shared_context_leaf=shared_context_leaf,
                context_join_leaf=context_join_leaf,
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                _critic_command(
                    payload,
                    iteration=iteration,
                    max_iterations=max_iterations,
                    recovery_final=recovery_final,
                ),
                ensure_ascii=False,
            ),
        },
    ]
    try:
        response = await _critic_atomic_chat(
            messages=messages,
            schema_name=schema_name,
            reasoning_effort=CRITIC_REASONING_EFFORT,
            temperature=0.1,
            transport_audit_checkpoint=transport_audit_checkpoint,
            transport_resume_lookup=transport_resume_lookup,
        )
    except OpenRouterResponseContractError as exc:
        incomplete_review, failure = _incomplete_transport_review(exc)
        if recovery_final:
            raise OpenRouterResponseContractError(
                "Final recovery critic primary response was incomplete or "
                f"unparseable ({failure}); compact repair is forbidden",
                result=exc.result,
            ) from exc
        if isinstance(exc, OpenRouterOutputLimitError):
            return await _recover_output_limited_review_by_decision(
                payload,
                incomplete_review,
                iteration=iteration,
                max_iterations=max_iterations,
                recovery_final=recovery_final,
                schema_name=schema_name,
                map_leaf=map_leaf,
                fragment_leaf=fragment_leaf,
                shared_context_leaf=shared_context_leaf,
                context_join_leaf=context_join_leaf,
                validate_schema=validate_schema,
                primary_usage=exc.result.usage,
                transport_audit_checkpoint=transport_audit_checkpoint,
                transport_resume_lookup=transport_resume_lookup,
            )
        repaired, repair_raw_text, repair_usage = await repair_analysis_review(
            payload,
            incomplete_review,
            iteration=iteration,
            validation_errors=[
                "Primary critic transport completed but its structured response "
                f"was unusable ({failure}): {exc}"
            ],
            recovery_final=recovery_final,
            audit_sink=audit_sink,
            transport_audit_checkpoint=transport_audit_checkpoint,
            transport_resume_lookup=transport_resume_lookup,
        )
        if (
            repaired.get("verdict") == "pass"
            and not _transport_repair_may_pass(incomplete_review)
        ):
            raise OpenRouterError(
                "Compact critic repair cannot promote an unparseable or "
                "non-passing primary response to pass"
            )
        raw_text = (
            json.dumps(
                {
                    "version": CRITIC_MAP_REDUCE_VERSION,
                    "primary_partial": incomplete_review,
                    "repair_raw_text": repair_raw_text,
                },
                ensure_ascii=False,
            )
            if map_leaf
            else repair_raw_text
        )
        return (
            (
                _validated_review_schema(repaired, owner=schema_name)
                if validate_schema
                else _normalize_review(repaired)
            ),
            raw_text,
            _merge_recovery_usage(
                exc.result.usage,
                repair_usage,
                recovered_from=failure,
            ),
        )
    if not isinstance(response.parsed, dict):
        raise OpenRouterError("Analysis critic returned no structured verdict")
    parsed = (
        _validated_review_schema(response.parsed, owner=schema_name)
        if validate_schema
        else _normalize_review(response.parsed)
    )
    return parsed, response.text, _critic_usage(response.usage)


def _critic_call_provenance(
    *,
    kind: str,
    index: int,
    input_value: dict[str, Any],
    raw_text: str,
    usage: dict[str, Any],
    verdict: str,
    lineage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "index": index,
        "input_sha256": _stable_sha256(input_value),
        "raw_response_sha256": hashlib.sha256(
            raw_text.encode("utf-8")
        ).hexdigest(),
        "raw_response_chars": len(raw_text),
        "verdict": verdict,
        "lineage": _json_safe(lineage or {}),
        "usage": _json_safe(usage),
    }


async def _emit_critic_call_audit(
    audit_sink: CriticCallAuditSink | None,
    *,
    attempt_id: str,
    iteration: int,
    kind: str,
    index: int,
    call_input: dict[str, Any],
    lineage: dict[str, Any],
    status: str,
    review: dict[str, Any] | None,
    raw_text: str,
    usage: dict[str, Any],
    provider_response_present: bool,
    error: BaseException | None,
    schema_name: str = "aiv_analysis_critic",
) -> None:
    """Durably hand off one provider-call outcome before later reduction."""

    if audit_sink is None:
        return
    descriptor = _critic_logical_call_descriptor(
        iteration=iteration,
        kind=kind,
        index=index,
        schema_name=schema_name,
        call_input=call_input,
        lineage=lineage,
    )
    expected_attempt_id = _critic_attempt_id(descriptor)
    if attempt_id != expected_attempt_id:
        raise OpenRouterError(
            "Critic audit attempt_id does not match the logical call digest"
        )
    event = {
        "version": CRITIC_CALL_AUDIT_VERSION,
        "critic_version": CRITIC_VERSION,
        "attempt_id": attempt_id,
        "iteration": iteration,
        "kind": kind,
        "index": index,
        "status": status,
        "model": CRITIC_MODEL,
        "schema_name": schema_name,
        "schema_sha256": descriptor["schema_sha256"],
        "logical_call_key": descriptor["logical_call_key"],
        "input": call_input,
        "input_sha256": _stable_sha256(call_input),
        "lineage": lineage,
        "lineage_sha256": descriptor["lineage_sha256"],
        "output": review,
        "output_sha256": (
            _stable_sha256(review) if isinstance(review, dict) else None
        ),
        "raw_text": raw_text,
        "raw_response_sha256": (
            hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
            if provider_response_present
            else None
        ),
        "raw_response_chars": len(raw_text),
        "provider_response_present": provider_response_present,
        "usage": usage,
        "error_type": type(error).__name__ if error is not None else None,
        "error_message": str(error) if error is not None else None,
    }
    # Saving the audit record is part of the provider-call contract. Shield it
    # so a sibling failure cannot cancel the only durable copy of paid output.
    await asyncio.shield(audit_sink(_json_safe(event)))


def _review_reduce_lineage_token(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _review_reduce_lineage_descriptor(values: list[Any]) -> dict[str, Any]:
    return {
        "source_count": len(values),
        "lineage_sha256": _stable_sha256(values),
        "first_source": values[0] if values else None,
        "last_source": values[-1] if values else None,
    }


def _verify_review_reduce_lineage(
    nodes: list[dict[str, Any]],
    *,
    expected_lineage: list[Any],
    owner: str,
) -> None:
    """Prove that every source is present exactly once at each tree level."""

    expected_tokens = [
        _review_reduce_lineage_token(value) for value in expected_lineage
    ]
    actual_lineage = [
        value
        for node in nodes
        for value in list(node.get("lineage") or [])
    ]
    actual_tokens = [
        _review_reduce_lineage_token(value) for value in actual_lineage
    ]
    duplicate = sorted(
        token for token in set(actual_tokens) if actual_tokens.count(token) > 1
    )
    if actual_tokens != expected_tokens or duplicate:
        missing = sorted(set(expected_tokens) - set(actual_tokens))
        unexpected = sorted(set(actual_tokens) - set(expected_tokens))
        raise OpenRouterError(
            f"{owner} refused incomplete hierarchical lineage: "
            f"missing={missing}, duplicate={duplicate}, unexpected={unexpected}, "
            f"ordered={actual_tokens == expected_tokens}"
        )


def _aggregate_provider_call_usage(
    calls: list[dict[str, Any]],
) -> dict[str, Any] | None:
    usages = [
        call["usage"]
        for call in calls
        if isinstance(call.get("usage"), dict)
    ]
    if not usages:
        return None
    output: dict[str, Any] = {}
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "reasoning_tokens",
        "cost",
    ):
        values = [
            usage.get(key)
            for usage in usages
            if isinstance(usage.get(key), (int, float))
            and not isinstance(usage.get(key), bool)
        ]
        if values:
            output[key] = sum(values)
    output["_aiv_critic_reduce_tree"] = {
        "provider_call_count": len(calls),
        "calls": [
            {
                "level": call["level"],
                "group_index": call["group_index"],
                "lineage": call["lineage"],
                "status": call["status"],
                "request_utf8_bytes": call["request_utf8_bytes"],
                "usage": call["usage"],
            }
            for call in calls
        ],
    }
    return _json_safe(output)


async def _hierarchical_review_reduce(
    *,
    source_nodes: list[dict[str, Any]],
    expected_lineage: list[Any],
    build_payload: Callable[
        [list[dict[str, Any]], int, int],
        dict[str, Any],
    ],
    system_prompt: str,
    schema_name_prefix: str,
    owner: str,
    audit_kind: str,
    iteration: int,
    recovery_final: bool,
    input_budget_bytes: int,
    context_envelope: dict[str, Any] | None = None,
    audit_sink: CriticCallAuditSink | None = None,
    transport_audit_checkpoint: AuditCheckpoint | None = None,
    transport_resume_lookup: TransportResumeLookup | None = None,
    collapse_answer_duplicates: bool = False,
    force_single_root_call: bool = True,
) -> dict[str, Any]:
    """Reduce an arbitrary number of reviews with bounded, proven fan-in.

    Provider calls only see a group that fits the current request budget.  The
    lineage union is owned and verified by code at every level.  If even two
    already-structured reviews cannot share one model window, the terminal
    union is deterministic rather than a length-dependent failure.  Each
    reducer verdict is one atomic provider observation; map/reduce expands
    input capacity, never output continuation.
    """

    if not source_nodes or not expected_lineage:
        raise OpenRouterError(f"{owner} requires at least one review source")
    current = [
        {
            "lineage": list(node.get("lineage") or []),
            "review": dict(node["review"]),
        }
        for node in source_nodes
    ]
    _verify_review_reduce_lineage(
        current,
        expected_lineage=expected_lineage,
        owner=owner,
    )
    original_reviews = [dict(node["review"]) for node in current]
    provider_calls: list[dict[str, Any]] = []
    tree_levels: list[dict[str, Any]] = []
    level = 0
    force_root = force_single_root_call and len(current) == 1

    while len(current) > 1 or force_root:
        groups: list[list[dict[str, Any]]] = []
        current_group: list[dict[str, Any]] = []
        for node in current:
            candidate = [*current_group, node]
            candidate_payload = build_payload(
                candidate,
                level,
                len(groups),
            )
            candidate_messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        candidate_payload,
                        ensure_ascii=False,
                    ),
                },
            ]
            candidate_bytes = (
                _critic_messages_physical_request_utf8_bytes(
                    messages=candidate_messages,
                    schema_name=(
                        f"{schema_name_prefix}_l{level}_g{len(groups)}"
                    ),
                    reasoning_effort=CRITIC_REASONING_EFFORT,
                    temperature=0.0,
                    context_envelope=context_envelope,
                )
                if isinstance(context_envelope, dict)
                else _json_utf8_bytes(candidate_messages)
            )
            if not current_group or candidate_bytes <= input_budget_bytes:
                current_group = candidate
                continue
            groups.append(current_group)
            current_group = [node]
        if current_group:
            groups.append(current_group)

        force_this_level = force_root
        force_root = False
        callable_groups = [
            (group_index, group)
            for group_index, group in enumerate(groups)
            if len(group) > 1 or force_this_level
        ]
        tree_levels.append(
            {
                "level": level,
                "input_node_count": len(current),
                "group_lineage": [
                    [
                        value
                        for node in group
                        for value in node["lineage"]
                    ]
                    for group in groups
                ],
                "provider_group_count": len(callable_groups),
            }
        )
        if not callable_groups:
            # Every pair is too large.  All findings remain available through
            # the code-owned union; recovery-final must not fail merely because
            # the corpus contains many or verbose child reviews.
            proposed = _critic_review_pass_seed(
                "Итог собран детерминированным объединением всех "
                "проверенных частей без потери lineage."
            )
            final_review = _merge_reviews_preserving_material_findings(
                original_reviews,
                proposed,
            )
            if collapse_answer_duplicates:
                final_review = _collapse_answer_overlap_duplicates(final_review)
            return {
                "review": final_review,
                "provider_calls": sorted(
                    provider_calls,
                    key=lambda call: (call["level"], call["group_index"]),
                ),
                "tree_levels": tree_levels,
                "status": "deterministic_terminal_union",
                "max_request_utf8_bytes": max(
                    (
                        int(call["request_utf8_bytes"])
                        for call in provider_calls
                    ),
                    default=0,
                ),
            }

        async def reduce_group(
            group_index: int,
            group: list[dict[str, Any]],
        ) -> dict[str, Any]:
            lineage = [
                value for node in group for value in node["lineage"]
            ]
            call_payload = build_payload(group, level, group_index)
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(call_payload, ensure_ascii=False),
                },
            ]
            request_bytes = (
                _critic_messages_physical_request_utf8_bytes(
                    messages=messages,
                    schema_name=(
                        f"{schema_name_prefix}_l{level}_g{group_index}"
                    ),
                    reasoning_effort=CRITIC_REASONING_EFFORT,
                    temperature=0.0,
                    context_envelope=context_envelope,
                )
                if isinstance(context_envelope, dict)
                else _json_utf8_bytes(messages)
            )
            if request_bytes > input_budget_bytes:
                # A singleton may itself be larger than the budget.  It cannot
                # be made smaller here without truncating a paid child review.
                return {
                    "lineage": lineage,
                    "review": _merge_reviews_preserving_material_findings(
                        [node["review"] for node in group],
                        _critic_review_pass_seed(
                            "Группа сохранена детерминированно: её нельзя "
                            "поместить в одно окно без обрезания."
                        ),
                    ),
                }
            schema_name = f"{schema_name_prefix}_l{level}_g{group_index}"
            call_lineage = {"sources": lineage, "tree_level": level}
            descriptor = _critic_logical_call_descriptor(
                iteration=iteration,
                kind=audit_kind,
                index=level * 100_000 + group_index,
                schema_name=schema_name,
                call_input=call_payload,
                lineage=call_lineage,
            )
            attempt_id = _critic_attempt_id(descriptor)
            cached = await _lookup_completed_critic_call(
                audit_sink,
                descriptor,
            )
            if cached is not None:
                proposed, raw_text, usage = cached
                provider_calls.append(
                    {
                        "kind": audit_kind,
                        "level": level,
                        "group_index": group_index,
                        "lineage": lineage,
                        "input": call_payload,
                        "raw_text": raw_text,
                        "usage": usage,
                        "status": "reused",
                        "request_utf8_bytes": request_bytes,
                    }
                )
                merged = _merge_reviews_preserving_material_findings(
                    [node["review"] for node in group],
                    proposed,
                )
                if collapse_answer_duplicates:
                    merged = _collapse_answer_overlap_duplicates(merged)
                return {"lineage": lineage, "review": merged}
            response: Any = None
            try:
                response = await _critic_atomic_chat(
                    messages=messages,
                    schema_name=schema_name,
                    reasoning_effort=CRITIC_REASONING_EFFORT,
                    temperature=0.0,
                    transport_audit_checkpoint=transport_audit_checkpoint,
                    transport_resume_lookup=transport_resume_lookup,
                )
                if not isinstance(response.parsed, dict):
                    raise OpenRouterError(
                        f"{owner} returned no structured verdict"
                    )
                proposed = _validated_review_schema(
                    response.parsed,
                    owner=(
                        f"{owner} level {level} group {group_index}"
                    ),
                )
            except BaseException as exc:
                result = getattr(exc, "result", None) or response
                raw_text = str(getattr(result, "text", "") or "")
                usage = (
                    _critic_usage(dict(getattr(result, "usage", {}) or {}))
                    if result is not None
                    else {}
                )
                await _emit_critic_call_audit(
                    audit_sink,
                    attempt_id=attempt_id,
                    iteration=iteration,
                    kind=audit_kind,
                    index=level * 100_000 + group_index,
                    call_input=call_payload,
                    lineage=call_lineage,
                    status=(
                        "cancelled"
                        if isinstance(exc, asyncio.CancelledError)
                        else "failed"
                    ),
                    review=None,
                    raw_text=raw_text,
                    usage=usage,
                    provider_response_present=result is not None,
                    error=exc,
                    schema_name=schema_name,
                )
                call_record = {
                    "kind": audit_kind,
                    "level": level,
                    "group_index": group_index,
                    "lineage": lineage,
                    "input": call_payload,
                    "raw_text": raw_text,
                    "usage": usage,
                    "status": "failed",
                    "request_utf8_bytes": request_bytes,
                }
                if (
                    isinstance(exc, OpenRouterResponseContractError)
                    and not recovery_final
                ):
                    provider_calls.append(call_record)
                    fallback = _merge_reviews_preserving_material_findings(
                        [node["review"] for node in group],
                        _incomplete_transport_review(exc)[0],
                    )
                    if collapse_answer_duplicates:
                        fallback = _collapse_answer_overlap_duplicates(fallback)
                    return {"lineage": lineage, "review": fallback}
                raise

            raw_text = response.text
            usage = _critic_usage(response.usage)
            await _emit_critic_call_audit(
                audit_sink,
                attempt_id=attempt_id,
                iteration=iteration,
                kind=audit_kind,
                index=level * 100_000 + group_index,
                call_input=call_payload,
                lineage=call_lineage,
                status="completed",
                review=proposed,
                raw_text=raw_text,
                usage=usage,
                provider_response_present=True,
                error=None,
                schema_name=schema_name,
            )
            provider_calls.append(
                {
                    "kind": audit_kind,
                    "level": level,
                    "group_index": group_index,
                    "lineage": lineage,
                    "input": call_payload,
                    "raw_text": raw_text,
                    "usage": usage,
                    "status": "completed",
                    "request_utf8_bytes": request_bytes,
                }
            )
            merged = _merge_reviews_preserving_material_findings(
                [node["review"] for node in group],
                proposed,
            )
            if collapse_answer_duplicates:
                merged = _collapse_answer_overlap_duplicates(merged)
            return {"lineage": lineage, "review": merged}

        tasks: list[asyncio.Task[dict[str, Any]]] = []
        passthrough: dict[int, dict[str, Any]] = {}
        callable_indexes = {index for index, _group in callable_groups}
        for group_index, group in enumerate(groups):
            if group_index not in callable_indexes:
                passthrough[group_index] = group[0]
                continue
            tasks.append(
                asyncio.create_task(
                    reduce_group(group_index, group),
                    name=(
                        f"aiv-critic-{audit_kind}-{iteration}-"
                        f"{level}-{group_index}"
                    ),
                )
            )
        try:
            outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.CancelledError:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        failures = [
            outcome
            for outcome in outcomes
            if isinstance(outcome, BaseException)
        ]
        if failures:
            first_failure = next(
                (
                    outcome
                    for outcome in failures
                    if not isinstance(outcome, asyncio.CancelledError)
                ),
                failures[0],
            )
            if isinstance(first_failure, OpenRouterResponseContractError):
                # Preserve the exact paid result for recovery-final callers.
                raise first_failure
            raise OpenRouterError(
                f"{owner} hierarchical reducer failed after durable sibling "
                f"audit: failures={len(failures)}"
            ) from first_failure
        reduced_by_index: dict[int, dict[str, Any]] = dict(passthrough)
        outcome_index = 0
        for group_index, _group in callable_groups:
            outcome = outcomes[outcome_index]
            outcome_index += 1
            if not isinstance(outcome, dict):
                raise OpenRouterError(
                    f"{owner} reducer produced an invalid task outcome"
                )
            reduced_by_index[group_index] = outcome
        current = [reduced_by_index[index] for index in range(len(groups))]
        _verify_review_reduce_lineage(
            current,
            expected_lineage=expected_lineage,
            owner=owner,
        )
        level += 1

    final_review = _merge_reviews_preserving_material_findings(
        original_reviews,
        current[0]["review"],
    )
    if collapse_answer_duplicates:
        final_review = _collapse_answer_overlap_duplicates(final_review)
    ordered_calls = sorted(
        provider_calls,
        key=lambda call: (call["level"], call["group_index"]),
    )
    return {
        "review": final_review,
        "provider_calls": ordered_calls,
        "tree_levels": tree_levels,
        "status": (
            "deterministic_terminal_union"
            if not ordered_calls
            else (
                "provider_reduced"
                if len(ordered_calls) == 1
                else "provider_hierarchical_reduced"
            )
        ),
        "max_request_utf8_bytes": max(
            (
                int(call["request_utf8_bytes"])
                for call in ordered_calls
            ),
            default=0,
        ),
    }


def _json_pointer_part(value: Any) -> str:
    return str(value).replace("~", "~0").replace("/", "~1")


def _shared_context_fact_priority(path: str) -> tuple[int, int, str]:
    """Put identity/scope rules before bulky narrative and metric evidence."""

    top_level = path.split("/", 2)[1] if path.startswith("/") else ""
    top_rank = {
        "site_profile": 0,
        "entity_catalog": 1,
        "attribution_owner_aliases": 2,
        "entity_attribution_aliases": 3,
        "metric_contract": 4,
        "deterministic_warnings": 5,
        "candidate_metrics": 6,
        "previous_policy_changes": 7,
        "raw_evidence_selection": 8,
    }.get(top_level, 9)
    field = path.rsplit("/", 1)[-1]
    field_rank = {
        "brand_name": 0,
        "canonical_name": 1,
        "target_aliases": 2,
        "brand_aliases": 3,
        "aliases": 4,
        "relationship": 5,
        "target_relationship": 6,
        "category": 7,
        "mention_policy": 8,
        "commercially_relevant": 9,
        "_profile_membership_confirmed": 10,
    }.get(field, 20)
    return top_rank, field_rank, path


def _shared_context_semantic_inventory(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Flatten the actual shared contract into deterministic atomic facts.

    Context-leaf model summaries are useful but cannot be the sole authority
    for entity attribution.  Every scalar is retained exactly here, including
    arbitrarily large narrative values.  Physical request partitioning happens
    later and may create any number of lossless fact units; the inventory never
    substitutes content with a digest or drops a late fact.
    """

    source = _shared_context_source(payload)
    canonical_source = json.dumps(
        source,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    )
    source_sha256 = hashlib.sha256(
        canonical_source.encode("utf-8")
    ).hexdigest()
    facts: list[dict[str, Any]] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            if not value:
                facts.append({"path": path, "value": {}})
                return
            for key in sorted(value, key=lambda item: str(item)):
                visit(
                    value[key],
                    path + "/" + _json_pointer_part(key),
                )
            return
        if isinstance(value, list):
            if not value:
                facts.append({"path": path, "value": []})
                return
            for index, child in enumerate(value):
                visit(child, path + f"/{index}")
            return
        safe_value = _json_safe(value)
        fact = {"path": path, "value": safe_value}
        facts.append(fact)

    for key in sorted(source):
        visit(source[key], "/" + _json_pointer_part(key))
    facts.sort(key=lambda fact: _shared_context_fact_priority(fact["path"]))
    return {
        "source_sha256": source_sha256,
        "fact_count": len(facts),
        "facts_sha256": _stable_sha256(facts),
        "facts": facts,
    }


def _shared_context_answer_digest(
    *,
    review: dict[str, Any],
    partition_manifest: dict[str, Any],
    semantic_inventory: dict[str, Any],
) -> dict[str, Any]:
    """Build a compact reduction receipt, never a substitute for raw facts.

    Exact facts are joined to answer/query leaves through lossless keyed shards.
    This receipt is intentionally content-addressed and compact so final
    reducers can attest the context-only verdict without repeating the corpus.
    """

    shared_manifest = dict(partition_manifest.get("shared_context") or {})
    source_sha256 = shared_manifest.get("source_sha256")
    if semantic_inventory.get("source_sha256") != source_sha256:
        raise OpenRouterError(
            "Shared-context semantic inventory does not match lossless source"
        )
    return {
        "version": CRITIC_MAP_REDUCE_VERSION,
        "status": "context_reduce_receipt_pending_answer_binding",
        "source": _shared_context_manifest_brief(shared_manifest),
        "semantic_context": {
            "source_sha256": semantic_inventory["source_sha256"],
            "fact_count": semantic_inventory["fact_count"],
            "facts_sha256": semantic_inventory["facts_sha256"],
            "delivery": "pending_lossless_answer_context_join",
        },
        "context_review": {
            "model_digest_sha256": _stable_sha256(review),
            "verdict": str(review.get("verdict") or "block"),
            "material_counts": {
                "anomalies": len(review.get("anomalies") or []),
                "policy_adjustments": len(
                    review.get("policy_adjustments") or []
                ),
                "acceptance_checks": len(
                    review.get("acceptance_checks") or []
                ),
            },
        },
    }


def _context_fact_unit_content_sha256(unit: dict[str, Any]) -> str:
    return _stable_sha256(
        {
            key: value
            for key, value in unit.items()
            if key != "unit_content_sha256"
        }
    )


def _build_context_fact_units(
    semantic_inventory: dict[str, Any],
    *,
    per_call_reserve_bytes: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create exact atomic/fractional units for every shared-context fact."""

    facts = semantic_inventory.get("facts")
    if not isinstance(facts, list):
        raise OpenRouterError("Shared-context semantic facts are missing")
    if semantic_inventory.get("fact_count") != len(facts) or (
        semantic_inventory.get("facts_sha256") != _stable_sha256(facts)
    ):
        raise OpenRouterError(
            "Shared-context semantic inventory count/digest is corrupt"
        )
    if not facts:
        raise OpenRouterError(
            "Shared-context semantic inventory has no attestable facts"
        )
    if (
        isinstance(per_call_reserve_bytes, bool)
        or not isinstance(per_call_reserve_bytes, int)
        or per_call_reserve_bytes <= 0
    ):
        raise OpenRouterError("Context-join reserve is invalid")

    # These are physical shard windows, never corpus limits. Small windows
    # merely create more units/calls. 256 is the lossless primitive's minimum.
    full_fact_bytes = max(256, per_call_reserve_bytes // 8)
    fragment_target_chars = max(
        256,
        min(1_024, per_call_reserve_bytes // 24),
    )
    fragment_overlap_chars = min(128, fragment_target_chars // 4)
    units: list[dict[str, Any]] = []
    fact_manifests: list[dict[str, Any]] = []
    for fact_index, raw_fact in enumerate(facts):
        if not isinstance(raw_fact, dict) or set(raw_fact) != {"path", "value"}:
            raise OpenRouterError(
                f"Shared-context fact {fact_index} has an invalid exact shape"
            )
        fact = _json_safe(raw_fact)
        fact_sha256 = _stable_sha256(fact)
        fact_id = f"fact:{fact_index}:{fact_sha256}"
        serialized = json.dumps(
            fact,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        serialized_bytes = len(serialized.encode("utf-8"))
        fact_unit_ids: list[str] = []
        if serialized_bytes <= full_fact_bytes:
            unit = {
                "mode": "complete_fact",
                "unit_id": fact_id,
                "fact_id": fact_id,
                "fact_index": fact_index,
                "fact_sha256": fact_sha256,
                "fact_utf8_bytes": serialized_bytes,
                "fact": fact,
            }
            unit["unit_content_sha256"] = _context_fact_unit_content_sha256(
                unit
            )
            units.append(unit)
            fact_unit_ids.append(fact_id)
            fact_manifests.append(
                {
                    "fact_id": fact_id,
                    "fact_index": fact_index,
                    "fact_sha256": fact_sha256,
                    "fact_utf8_bytes": serialized_bytes,
                    "mode": "complete_fact",
                    "unit_ids": fact_unit_ids,
                }
            )
            continue

        text_units, text_manifest = split_lossless_text(
            serialized,
            document_id=f"critic-shared-fact-{fact_index}-{fact_sha256[:16]}",
            target_chars=fragment_target_chars,
            context_overlap_chars=fragment_overlap_chars,
        )
        lossless_manifest = text_manifest.as_dict()
        if verify_units(text_units, lossless_manifest) != serialized:
            raise OpenRouterError(
                f"Shared-context fact {fact_index} failed lossless partition"
            )
        for text_unit in text_units:
            unit = {
                "mode": "fact_fragment",
                "unit_id": text_unit.unit_id,
                "fact_id": fact_id,
                "fact_index": fact_index,
                "fact_path": fact["path"],
                "fact_sha256": fact_sha256,
                "fact_utf8_bytes": serialized_bytes,
                "fact_json_fragment": text_unit.context_text,
                "fragment": {
                    "source_sha256": lossless_manifest["source_sha256"],
                    "source_chars": lossless_manifest["source_chars"],
                    "source_utf8_bytes": lossless_manifest[
                        "source_utf8_bytes"
                    ],
                    "unit_id": text_unit.unit_id,
                    "unit_index": text_unit.index,
                    "unit_count": lossless_manifest["unit_count"],
                    "core_start_char": text_unit.start_char,
                    "core_end_char": text_unit.end_char,
                    "core_sha256": text_unit.sha256,
                    "context_start_char": text_unit.context_start_char,
                    "context_end_char": text_unit.context_end_char,
                    "context_sha256": text_unit.context_sha256,
                    "core_start_in_context": (
                        text_unit.core_start_in_context
                    ),
                    "core_end_in_context": text_unit.core_end_in_context,
                    "overlap_counts_toward_core_coverage": False,
                },
            }
            unit["unit_content_sha256"] = _context_fact_unit_content_sha256(
                unit
            )
            units.append(unit)
            fact_unit_ids.append(text_unit.unit_id)
        fact_manifests.append(
            {
                "fact_id": fact_id,
                "fact_index": fact_index,
                "fact_sha256": fact_sha256,
                "fact_utf8_bytes": serialized_bytes,
                "mode": "lossless_fact_fragments",
                "unit_ids": fact_unit_ids,
                "lossless_manifest": lossless_manifest,
                "lossless_manifest_sha256": _stable_sha256(lossless_manifest),
            }
        )

    unit_ids = [str(unit["unit_id"]) for unit in units]
    manifest = {
        "version": CRITIC_MAP_REDUCE_VERSION,
        "mode": "lossless_context_fact_units",
        "source_sha256": semantic_inventory["source_sha256"],
        "fact_count": len(facts),
        "facts_sha256": semantic_inventory["facts_sha256"],
        "unit_count": len(units),
        "unit_ids": unit_ids,
        "unit_ids_sha256": _stable_sha256(unit_ids),
        "fact_manifests": fact_manifests,
        "missing_fact_ids": [],
        "duplicate_unit_ids": [],
        "exact_fact_accounting": True,
    }
    _validate_context_fact_units(units, manifest, semantic_inventory)
    return units, manifest


def _validate_context_fact_units(
    units: list[dict[str, Any]],
    manifest: dict[str, Any],
    semantic_inventory: dict[str, Any],
) -> None:
    """Fail closed unless every exact inventory fact is fully represented."""

    facts = semantic_inventory.get("facts")
    if not isinstance(facts, list):
        raise OpenRouterError("Context fact inventory is missing")
    if (
        manifest.get("source_sha256")
        != semantic_inventory.get("source_sha256")
        or manifest.get("fact_count") != len(facts)
        or manifest.get("facts_sha256") != _stable_sha256(facts)
    ):
        raise OpenRouterError("Context fact manifest targets another inventory")
    actual_ids = [str(unit.get("unit_id") or "") for unit in units]
    expected_ids = [str(value) for value in manifest.get("unit_ids") or []]
    duplicates = sorted(
        unit_id for unit_id in set(actual_ids) if actual_ids.count(unit_id) > 1
    )
    if (
        actual_ids != expected_ids
        or duplicates
        or manifest.get("unit_count") != len(actual_ids)
        or manifest.get("unit_ids_sha256") != _stable_sha256(actual_ids)
    ):
        raise OpenRouterError(
            "Context fact unit coverage is missing, duplicated, or reordered"
        )
    units_by_id = {str(unit["unit_id"]): unit for unit in units}
    fact_manifests = manifest.get("fact_manifests")
    if not isinstance(fact_manifests, list) or len(fact_manifests) != len(facts):
        raise OpenRouterError("Context fact manifests are incomplete")
    covered_ids: list[str] = []
    for fact_index, (fact, fact_manifest) in enumerate(
        zip(facts, fact_manifests, strict=True)
    ):
        if not isinstance(fact_manifest, dict):
            raise OpenRouterError("Context fact manifest entry is corrupt")
        expected_fact_sha256 = _stable_sha256(fact)
        fact_id = f"fact:{fact_index}:{expected_fact_sha256}"
        unit_ids = [str(value) for value in fact_manifest.get("unit_ids") or []]
        if (
            fact_manifest.get("fact_id") != fact_id
            or fact_manifest.get("fact_index") != fact_index
            or fact_manifest.get("fact_sha256") != expected_fact_sha256
            or not unit_ids
        ):
            raise OpenRouterError("Context fact identity/lineage is corrupt")
        fact_units = [units_by_id.get(unit_id) for unit_id in unit_ids]
        if any(unit is None for unit in fact_units):
            raise OpenRouterError("Context fact lineage references a missing unit")
        for unit in fact_units:
            if not isinstance(unit, dict) or unit.get(
                "unit_content_sha256"
            ) != _context_fact_unit_content_sha256(unit):
                raise OpenRouterError("Context fact unit content was mutated")
            if (
                unit.get("fact_id") != fact_id
                or unit.get("fact_index") != fact_index
                or unit.get("fact_sha256") != expected_fact_sha256
            ):
                raise OpenRouterError("Context fact unit cross-binding is corrupt")
        if fact_manifest.get("mode") == "complete_fact":
            if len(fact_units) != 1 or fact_units[0].get("fact") != fact:
                raise OpenRouterError("Complete context fact content was mutated")
        elif fact_manifest.get("mode") == "lossless_fact_fragments":
            reconstructed_parts: list[str] = []
            for unit_index, unit in enumerate(fact_units):
                fragment = unit.get("fragment")
                context = unit.get("fact_json_fragment")
                if (
                    not isinstance(fragment, dict)
                    or not isinstance(context, str)
                    or fragment.get("unit_index") != unit_index
                    or fragment.get("unit_id") != unit.get("unit_id")
                    or fragment.get("context_sha256")
                    != hashlib.sha256(context.encode("utf-8")).hexdigest()
                ):
                    raise OpenRouterError("Context fact fragment was mutated")
                start = fragment.get("core_start_in_context")
                end = fragment.get("core_end_in_context")
                if (
                    isinstance(start, bool)
                    or not isinstance(start, int)
                    or isinstance(end, bool)
                    or not isinstance(end, int)
                    or not 0 <= start <= end <= len(context)
                ):
                    raise OpenRouterError("Context fact fragment core is invalid")
                core = context[start:end]
                if hashlib.sha256(core.encode("utf-8")).hexdigest() != fragment.get(
                    "core_sha256"
                ):
                    raise OpenRouterError("Context fact fragment core was mutated")
                reconstructed_parts.append(core)
            reconstructed = "".join(reconstructed_parts)
            expected_serialized = json.dumps(
                fact,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            if reconstructed != expected_serialized:
                raise OpenRouterError(
                    "Context fact fragments do not reconstruct exact content"
                )
        else:
            raise OpenRouterError("Context fact manifest mode is unsupported")
        covered_ids.extend(unit_ids)
    if covered_ids != actual_ids:
        raise OpenRouterError(
            "Context fact manifests do not cover every unit exactly once"
        )


def _context_join_payload(
    base_payload: dict[str, Any],
    *,
    base_identity: dict[str, Any],
    fact_units: list[dict[str, Any]],
    fact_manifest: dict[str, Any],
    context_receipt: dict[str, Any],
    shard_index: int,
    shard_count: int,
) -> dict[str, Any]:
    unit_ids = [str(unit["unit_id"]) for unit in fact_units]
    binding_core = {
        "version": CRITIC_MAP_REDUCE_VERSION,
        "mode": "answer_context_joint_shard",
        "base_leaf_id": base_identity["base_leaf_id"],
        "base_kind": base_identity["base_kind"],
        "base_index": base_identity["base_index"],
        "base_payload_sha256": base_identity["base_payload_sha256"],
        "assigned_answer_ids": base_identity["assigned_answer_ids"],
        "answer_query_bindings": base_identity["answer_query_bindings"],
        "source_sha256": fact_manifest["source_sha256"],
        "facts_sha256": fact_manifest["facts_sha256"],
        "fact_count": fact_manifest["fact_count"],
        "complete_unit_count": fact_manifest["unit_count"],
        "complete_unit_ids_sha256": fact_manifest["unit_ids_sha256"],
        "context_shard_index": shard_index,
        "context_shard_count": shard_count,
        "assigned_context_unit_ids": unit_ids,
        "assigned_context_unit_count": len(unit_ids),
        "assigned_context_units_sha256": _stable_sha256(fact_units),
    }
    return {
        **base_payload,
        "shared_context_receipt": context_receipt,
        "critic_context_binding": {
            **binding_core,
            "binding_sha256": _stable_sha256(binding_core),
        },
        "shared_context_facts": fact_units,
    }


def _answer_query_bindings(
    base_payload: dict[str, Any],
    *,
    assigned_answer_ids: list[int],
) -> list[dict[str, Any]]:
    """Return explicit query lineage for every answer owned by a base leaf."""

    answers = base_payload.get("answers")
    if not isinstance(answers, list):
        raise OpenRouterError("Context-join base has no answer/query records")
    bindings: list[dict[str, Any]] = []
    actual_answer_ids: list[int] = []
    for answer in answers:
        if not isinstance(answer, dict):
            raise OpenRouterError("Context-join answer/query record is corrupt")
        answer_id = answer.get("answer_id")
        if isinstance(answer_id, bool) or not isinstance(answer_id, int):
            raise OpenRouterError("Context-join answer has no stable integer id")
        prompt_id = answer.get("prompt_id")
        prompt_key = answer.get("prompt_key")
        scenario = answer.get("scenario")
        if (
            isinstance(prompt_id, bool)
            or not isinstance(prompt_id, int)
            or not isinstance(prompt_key, str)
            or not prompt_key
            or not isinstance(scenario, str)
            or not scenario
        ):
            raise OpenRouterError(
                "Context-join answer has incomplete query lineage"
            )
        actual_answer_ids.append(answer_id)
        query_core = {
            "answer_id": answer_id,
            "prompt_id": prompt_id,
            "prompt_key": prompt_key,
            "scenario": scenario,
            "scenario_role": answer.get("scenario_role"),
            "intent_class": answer.get("intent_class"),
        }
        bindings.append(
            {
                **query_core,
                "query_binding_sha256": _stable_sha256(query_core),
            }
        )
    if actual_answer_ids != assigned_answer_ids:
        raise OpenRouterError(
            "Context-join answer/query lineage differs from assigned answers"
        )
    return bindings


def _validate_context_join_physical_preflight(task: dict[str, Any]) -> None:
    """Reconstruct the exact provider body and reject altered plan receipts."""

    payload = task.get("payload")
    preflight = task.get("physical_preflight")
    if not isinstance(payload, dict) or not isinstance(preflight, dict):
        raise OpenRouterError("Context-join physical preflight is missing")
    request_payload = preflight.get("request_payload")
    if not isinstance(request_payload, dict):
        raise OpenRouterError("Context-join physical request body is missing")
    iteration = task.get("iteration")
    max_iterations = task.get("max_iterations")
    recovery_final = task.get("recovery_final")
    schema_name = task.get("schema_name")
    if (
        isinstance(iteration, bool)
        or not isinstance(iteration, int)
        or isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or not isinstance(recovery_final, bool)
        or not isinstance(schema_name, str)
        or not schema_name
    ):
        raise OpenRouterError("Context-join physical request contract is corrupt")
    effective_max = preflight.get("effective_max_completion_tokens")
    if isinstance(effective_max, bool) or not isinstance(effective_max, int):
        raise OpenRouterError("Context-join physical output window is invalid")
    expected: dict[str, Any] = {
        "model": CRITIC_MODEL,
        "messages": [
            {
                "role": "system",
                "content": _critic_system_prompt(
                    recovery_final=recovery_final,
                    map_leaf=True,
                    fragment_leaf=task.get("base_kind") == "fragment_leaf",
                    context_join_leaf=True,
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    _critic_command(
                        payload,
                        iteration=iteration,
                        max_iterations=max_iterations,
                        recovery_final=recovery_final,
                    ),
                    ensure_ascii=False,
                ),
            },
        ],
        "temperature": 0.1,
        "reasoning": {
            "effort": CRITIC_REASONING_EFFORT,
            "exclude": True,
        },
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": CRITIC_SCHEMA,
            },
        },
        "max_completion_tokens": effective_max,
    }
    policy_fields, _policy = web_request_policy(
        model=CRITIC_MODEL,
        policy="forbidden",
    )
    expected.update(policy_fields)
    if request_payload != expected:
        raise OpenRouterError(
            "Context-join physical request differs from its exact contract"
        )
    serialized = json.dumps(
        request_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(serialized).hexdigest()
    if (
        preflight.get("fits_model_envelope") is not True
        or preflight.get("request_utf8_bytes") != len(serialized)
        or preflight.get("request_sha256") != digest
        or task.get("physical_request_utf8_bytes") != len(serialized)
        or task.get("physical_request_sha256") != digest
        or task.get("requested_output_tokens") != effective_max
    ):
        raise OpenRouterError(
            "Context-join physical request receipt is missing or tampered"
        )


def _context_fact_units_relevant_to_base(
    *,
    fact_units: list[dict[str, Any]],
    semantic_inventory: dict[str, Any],
    base_payload: dict[str, Any],
    assigned_answer_ids: list[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join only code-grounded relevant facts to one answer/query leaf.

    The complete inventory has already been read by the independently reduced
    shared-context tree. Repeating every exact fact beside every raw answer was
    a paid Cartesian product. This local inverted index keeps global identity
    rules, joins entity records on literal canonical names/aliases, and joins
    warning/policy/evidence records on explicit answer ids. Facts that have no
    such relation remain covered by the shared-context receipt; their absence
    from this leaf is never semantic evidence of absence.
    """

    facts = semantic_inventory.get("facts")
    if not isinstance(facts, list):
        raise OpenRouterError("Context relevance index has no semantic facts")
    base_text = json.dumps(
        base_payload,
        ensure_ascii=False,
        sort_keys=True,
    ).casefold()
    assigned = set(assigned_answer_ids)

    entity_groups: dict[int, list[int]] = {}
    scoped_groups: dict[tuple[str, int], list[int]] = {}
    for fact_index, fact in enumerate(facts):
        path = str(fact.get("path") or "") if isinstance(fact, dict) else ""
        entity_match = re.match(
            r"^/entity_catalog/entities/(\d+)(?:/|$)", path
        )
        if entity_match:
            entity_groups.setdefault(int(entity_match.group(1)), []).append(
                fact_index
            )
        scoped_match = re.match(
            r"^/(deterministic_warnings|previous_policy_changes|raw_evidence_selection)/(\d+)(?:/|$)",
            path,
        )
        if scoped_match:
            scoped_groups.setdefault(
                (scoped_match.group(1), int(scoped_match.group(2))), []
            ).append(fact_index)

    selected_fact_indices: set[int] = set()
    selection_reasons: dict[int, str] = {}

    def select(index: int, reason: str) -> None:
        selected_fact_indices.add(index)
        selection_reasons.setdefault(index, reason)

    always_exact = {
        "/site_profile/brand_name",
        "/site_profile/site_type",
        "/site_profile/category",
        "/site_profile/market",
        "/site_profile/business_model",
    }
    always_prefixes = (
        "/site_profile/brand_aliases/",
        "/entity_catalog/target_",
        "/attribution_owner_aliases/",
    )
    for fact_index, fact in enumerate(facts):
        if not isinstance(fact, dict):
            continue
        path = str(fact.get("path") or "")
        if path in always_exact or path.startswith(always_prefixes):
            select(fact_index, "global_identity_contract")

    for group_indices in entity_groups.values():
        literal_values = [
            str(facts[index].get("value") or "").strip().casefold()
            for index in group_indices
            if isinstance(facts[index].get("value"), str)
            and len(str(facts[index].get("value") or "").strip()) >= 3
            and any(
                marker in str(facts[index].get("path") or "")
                for marker in ("canonical_name", "/aliases/", "/name")
            )
        ]
        if any(value in base_text for value in literal_values):
            for fact_index in group_indices:
                select(fact_index, "literal_entity_binding")

    for group_indices in scoped_groups.values():
        scoped_answer_ids = {
            int(facts[index]["value"])
            for index in group_indices
            if isinstance(facts[index].get("value"), int)
            and not isinstance(facts[index].get("value"), bool)
            and "/answer_ids/" in str(facts[index].get("path") or "")
        }
        if not scoped_answer_ids or scoped_answer_ids & assigned:
            for fact_index in group_indices:
                select(fact_index, "explicit_answer_id_binding")

    # A malformed/minimal test payload may have none of the known identity
    # paths. Keep one exact fact so every answer leaf still has an independently
    # inspectable context binding instead of trusting a digest alone.
    if not selected_fact_indices and facts:
        select(0, "minimum_exact_context_anchor")

    selected_units = [
        unit
        for unit in fact_units
        if int(unit.get("fact_index") or 0) in selected_fact_indices
    ]
    selected_unit_ids = [str(unit["unit_id"]) for unit in selected_units]
    return selected_units, {
        "version": CRITIC_MAP_REDUCE_VERSION,
        "mode": "code_owned_relevance_index",
        "selected_fact_indices": sorted(selected_fact_indices),
        "selected_fact_count": len(selected_fact_indices),
        "selected_unit_ids": selected_unit_ids,
        "selected_unit_ids_sha256": _stable_sha256(selected_unit_ids),
        "selection_reasons": {
            str(index): selection_reasons[index]
            for index in sorted(selection_reasons)
        },
        "shared_context_only_fact_count": len(facts) - len(selected_fact_indices),
    }


def _build_context_join_tasks(
    *,
    base_entries: list[dict[str, Any]],
    semantic_inventory: dict[str, Any],
    context_receipt: dict[str, Any],
    iteration: int,
    max_iterations: int,
    recovery_final: bool,
    input_budget_bytes: int,
    context_envelope: dict[str, Any],
    per_call_reserve_bytes: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build bounded relevance-indexed answer/context joins.

    Every exact fact is covered once by the shared-context tree. Only facts
    with a code-grounded identity/entity/answer-id relationship are repeated
    beside a raw answer. This preserves inspectable evidence where a joint
    decision is possible without paying for answer x every-fact.
    """

    if not base_entries:
        raise OpenRouterError("Context-join requires at least one answer leaf")

    fact_units, fact_manifest = _build_context_fact_units(
        semantic_inventory,
        per_call_reserve_bytes=per_call_reserve_bytes,
    )
    tasks: list[dict[str, Any]] = []
    base_manifests: list[dict[str, Any]] = []
    global_index = 0
    for base_ordinal, entry in enumerate(base_entries):
        raw_payload = entry.get("payload")
        if not isinstance(raw_payload, dict):
            raise OpenRouterError("Context-join base payload is invalid")
        base_payload = {
            key: value
            for key, value in raw_payload.items()
            if key not in {"shared_context_digest", "shared_context_manifest"}
        }
        assigned_answer_ids = [
            int(value)
            for value in base_payload.get("critic_map_partition", {}).get(
                "assigned_answer_ids", []
            )
        ]
        if (
            not assigned_answer_ids
            or len(assigned_answer_ids) != len(set(assigned_answer_ids))
        ):
            raise OpenRouterError(
                "Context-join base has no answers or duplicate answer ids"
            )
        answer_query_bindings = _answer_query_bindings(
            base_payload,
            assigned_answer_ids=assigned_answer_ids,
        )
        base_kind = str(entry.get("base_kind") or "")
        if base_kind not in {"leaf", "fragment_leaf"}:
            raise OpenRouterError("Context-join base kind is unsupported")
        base_index = int(entry.get("base_index") or 0)
        base_identity_core = {
            "base_kind": base_kind,
            "base_index": base_index,
            "assigned_answer_ids": assigned_answer_ids,
            "answer_query_bindings": answer_query_bindings,
            "base_payload_sha256": _stable_sha256(base_payload),
        }
        base_identity = {
            **base_identity_core,
            "base_leaf_id": "base:" + _stable_sha256(base_identity_core),
        }
        if any(
            manifest.get("base_leaf_id") == base_identity["base_leaf_id"]
            for manifest in base_manifests
        ):
            raise OpenRouterError("Context-join base identity is ambiguous")
        relevant_fact_units, relevance_manifest = (
            _context_fact_units_relevant_to_base(
                fact_units=fact_units,
                semantic_inventory=semantic_inventory,
                base_payload=base_payload,
                assigned_answer_ids=assigned_answer_ids,
            )
        )
        grouped_units: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        for unit in relevant_fact_units:
            candidate = [*current, unit]
            provisional = _context_join_payload(
                base_payload,
                base_identity=base_identity,
                fact_units=candidate,
                fact_manifest=fact_manifest,
                context_receipt=context_receipt,
                shard_index=len(grouped_units),
                shard_count=999_999_999,
            )
            schema_name = (
                f"aiv_analysis_critic_{iteration}_context_join_"
                f"{base_ordinal}_{len(grouped_units)}"
            )
            preflight = _critic_physical_request_preflight(
                provisional,
                iteration=iteration,
                max_iterations=max_iterations,
                recovery_final=recovery_final,
                schema_name=schema_name,
                context_envelope=context_envelope,
                map_leaf=True,
                fragment_leaf=base_kind == "fragment_leaf",
                context_join_leaf=True,
            )
            requested_output = context_envelope.get("max_completion_tokens")
            if not isinstance(requested_output, int) or isinstance(
                requested_output, bool
            ):
                requested_output = context_envelope.get(
                    "reserved_output_tokens"
                )
            full_output_preserved = (
                not isinstance(requested_output, int)
                or preflight["effective_max_completion_tokens"]
                == requested_output
            )
            if (
                preflight["request_utf8_bytes"] <= input_budget_bytes
                and full_output_preserved
            ):
                current = candidate
                continue
            if not current:
                raise OpenRouterError(
                    "One minimum lossless context fact unit cannot share the "
                    "critic model window with its bound answer/query leaf: "
                    f"base_leaf_id={base_identity['base_leaf_id']}, "
                    f"unit_id={unit['unit_id']}, "
                    f"request_bytes={preflight['request_utf8_bytes']}, "
                    f"budget_bytes={input_budget_bytes}"
                )
            grouped_units.append(current)
            current = [unit]
        if current:
            grouped_units.append(current)

        base_task_ids: list[str] = []
        covered_unit_ids: list[str] = []
        for shard_index, group in enumerate(grouped_units):
            payload = _context_join_payload(
                base_payload,
                base_identity=base_identity,
                fact_units=group,
                fact_manifest=fact_manifest,
                context_receipt=context_receipt,
                shard_index=shard_index,
                shard_count=len(grouped_units),
            )
            schema_name = (
                f"aiv_analysis_critic_{iteration}_context_join_"
                f"{base_ordinal}_{shard_index}"
            )
            preflight = _critic_physical_request_preflight(
                payload,
                iteration=iteration,
                max_iterations=max_iterations,
                recovery_final=recovery_final,
                schema_name=schema_name,
                context_envelope=context_envelope,
                map_leaf=True,
                fragment_leaf=base_kind == "fragment_leaf",
                context_join_leaf=True,
            )
            requested_output = context_envelope.get("max_completion_tokens")
            if not isinstance(requested_output, int) or isinstance(
                requested_output, bool
            ):
                requested_output = context_envelope.get(
                    "reserved_output_tokens"
                )
            if (
                preflight.get("fits_model_envelope") is not True
                or preflight["request_utf8_bytes"] > input_budget_bytes
                or (
                    isinstance(requested_output, int)
                    and preflight["effective_max_completion_tokens"]
                    != requested_output
                )
            ):
                raise OpenRouterError(
                    "Final context-join physical request does not preserve the "
                    "exact model input/output envelope"
                )
            task_core = {
                "global_index": global_index,
                "iteration": iteration,
                "max_iterations": max_iterations,
                "recovery_final": recovery_final,
                "base_leaf_id": base_identity["base_leaf_id"],
                "base_kind": base_kind,
                "base_index": base_index,
                "assigned_answer_ids": assigned_answer_ids,
                "answer_query_bindings": answer_query_bindings,
                "context_shard_index": shard_index,
                "context_shard_count": len(grouped_units),
                "context_unit_ids": [str(unit["unit_id"]) for unit in group],
                "schema_name": schema_name,
                "payload_sha256": _stable_sha256(payload),
                "physical_request_sha256": preflight["request_sha256"],
                "physical_request_utf8_bytes": preflight[
                    "request_utf8_bytes"
                ],
                "requested_output_tokens": requested_output,
            }
            task_id = "context-join:" + _stable_sha256(task_core)
            task = {
                **task_core,
                "task_id": task_id,
                "payload": payload,
                "physical_preflight": preflight,
            }
            _validate_context_join_physical_preflight(task)
            tasks.append(task)
            base_task_ids.append(task_id)
            covered_unit_ids.extend(task_core["context_unit_ids"])
            global_index += 1
        expected_unit_ids = [
            str(value) for value in relevance_manifest["selected_unit_ids"]
        ]
        if covered_unit_ids != expected_unit_ids or len(covered_unit_ids) != len(
            set(covered_unit_ids)
        ):
            raise OpenRouterError(
                "Context-join planner failed exact per-answer fact coverage"
            )
        base_manifests.append(
            {
                **base_identity,
                "relevance_manifest": relevance_manifest,
                "relevance_manifest_sha256": _stable_sha256(
                    relevance_manifest
                ),
                "expected_context_unit_ids": expected_unit_ids,
                "task_ids": base_task_ids,
                "task_count": len(base_task_ids),
                "covered_unit_ids": covered_unit_ids,
                "covered_unit_ids_sha256": _stable_sha256(covered_unit_ids),
                "exact_relevant_unit_accounting": True,
            }
        )

    all_unit_ids = [str(value) for value in fact_manifest["unit_ids"]]
    joined_unit_ids = {
        unit_id
        for base in base_manifests
        for unit_id in base["expected_context_unit_ids"]
    }
    shared_context_only_unit_ids = [
        unit_id for unit_id in all_unit_ids if unit_id not in joined_unit_ids
    ]
    attached_unit_count = sum(
        len(base["expected_context_unit_ids"]) for base in base_manifests
    )
    manifest = {
        "version": CRITIC_MAP_REDUCE_VERSION,
        "mode": "relevance_indexed_answer_context_join",
        "fact_manifest": fact_manifest,
        "fact_manifest_sha256": _stable_sha256(fact_manifest),
        "base_count": len(base_manifests),
        "base_manifests": base_manifests,
        "task_count": len(tasks),
        "task_ids": [str(task["task_id"]) for task in tasks],
        "task_ids_sha256": _stable_sha256(
            [str(task["task_id"]) for task in tasks]
        ),
        "missing_task_ids": [],
        "duplicate_task_ids": [],
        "shared_context_only_unit_ids": shared_context_only_unit_ids,
        "shared_context_only_unit_ids_sha256": _stable_sha256(
            shared_context_only_unit_ids
        ),
        "joined_unique_unit_ids": [
            unit_id for unit_id in all_unit_ids if unit_id in joined_unit_ids
        ],
        "attached_relevant_unit_count": attached_unit_count,
        "naive_cross_product_unit_count": len(base_manifests) * len(all_unit_ids),
        "avoided_cross_product_unit_attachments": (
            len(base_manifests) * len(all_unit_ids) - attached_unit_count
        ),
        "paid_task_cost_bound": {
            "kind": "physical_input_packing_of_relevant_units_only",
            "base_count": len(base_manifests),
            "relevant_unit_attachments": attached_unit_count,
            "task_count": len(tasks),
            "task_count_lte_relevant_unit_attachments": (
                len(tasks) <= attached_unit_count
            ),
        },
        "exact_global_fact_accounting": True,
    }
    _validate_context_join_coverage(tasks, manifest)
    return tasks, manifest


def _validate_context_join_coverage(
    tasks: list[dict[str, Any]],
    manifest: dict[str, Any],
    results: list[dict[str, Any]] | None = None,
) -> None:
    """Verify plan/result lineage before any compact receipt is trusted."""

    fact_manifest = manifest.get("fact_manifest")
    if (
        not isinstance(fact_manifest, dict)
        or manifest.get("fact_manifest_sha256")
        != _stable_sha256(fact_manifest)
    ):
        raise OpenRouterError("Context-join fact manifest is missing or tampered")
    expected_task_ids = [str(value) for value in manifest.get("task_ids") or []]
    actual_task_ids = [str(task.get("task_id") or "") for task in tasks]
    if (
        actual_task_ids != expected_task_ids
        or len(actual_task_ids) != len(set(actual_task_ids))
        or manifest.get("task_count") != len(actual_task_ids)
        or manifest.get("task_ids_sha256") != _stable_sha256(actual_task_ids)
    ):
        raise OpenRouterError(
            "Context-join task lineage is missing, duplicated, or reordered"
        )
    task_by_id = {str(task["task_id"]): task for task in tasks}
    base_manifests = manifest.get("base_manifests")
    if (
        not isinstance(base_manifests, list)
        or not base_manifests
        or manifest.get("base_count") != len(base_manifests)
    ):
        raise OpenRouterError("Context-join base manifest is missing")
    base_ids = [
        str(base.get("base_leaf_id") or "")
        for base in base_manifests
        if isinstance(base, dict)
    ]
    if len(base_ids) != len(base_manifests) or len(base_ids) != len(set(base_ids)):
        raise OpenRouterError("Context-join base identities are ambiguous")
    all_fact_unit_ids = [
        str(value) for value in fact_manifest.get("unit_ids") or []
    ]
    joined_unique_ids = {
        str(value)
        for base in base_manifests
        if isinstance(base, dict)
        for value in base.get("expected_context_unit_ids") or []
    }
    expected_joined_order = [
        unit_id for unit_id in all_fact_unit_ids if unit_id in joined_unique_ids
    ]
    expected_context_only = [
        unit_id for unit_id in all_fact_unit_ids if unit_id not in joined_unique_ids
    ]
    if (
        manifest.get("mode") != "relevance_indexed_answer_context_join"
        or manifest.get("exact_global_fact_accounting") is not True
        or manifest.get("joined_unique_unit_ids") != expected_joined_order
        or manifest.get("shared_context_only_unit_ids") != expected_context_only
        or manifest.get("shared_context_only_unit_ids_sha256")
        != _stable_sha256(expected_context_only)
        or set(expected_joined_order) & set(expected_context_only)
        or sorted([*expected_joined_order, *expected_context_only])
        != sorted(all_fact_unit_ids)
    ):
        raise OpenRouterError(
            "Context-join relevance manifest lost global fact coverage"
        )
    for global_index, task in enumerate(tasks):
        task_core = {
            key: value
            for key, value in task.items()
            if key not in {"task_id", "payload", "physical_preflight"}
        }
        if (
            task.get("global_index") != global_index
            or task.get("task_id")
            != "context-join:" + _stable_sha256(task_core)
        ):
            raise OpenRouterError("Context-join task identity was mutated")
    covered_tasks: list[str] = []
    for base_manifest in base_manifests:
        if not isinstance(base_manifest, dict):
            raise OpenRouterError("Context-join base manifest is corrupt")
        base_task_ids = [str(value) for value in base_manifest.get("task_ids") or []]
        base_tasks = [task_by_id.get(task_id) for task_id in base_task_ids]
        if any(task is None for task in base_tasks):
            raise OpenRouterError("Context-join base references a missing task")
        relevance_manifest = base_manifest.get("relevance_manifest")
        if (
            not isinstance(relevance_manifest, dict)
            or base_manifest.get("relevance_manifest_sha256")
            != _stable_sha256(relevance_manifest)
        ):
            raise OpenRouterError(
                "Context-join base relevance manifest is missing or tampered"
            )
        expected_units = [
            str(value)
            for value in base_manifest.get("expected_context_unit_ids") or []
        ]
        actual_units = [
            str(unit_id)
            for task in base_tasks
            for unit_id in task.get("context_unit_ids") or []
        ]
        if (
            base_manifest.get("task_count") != len(base_task_ids)
            or len(base_task_ids) != len(set(base_task_ids))
            or actual_units != expected_units
            or len(actual_units) != len(set(actual_units))
            or base_manifest.get("covered_unit_ids") != actual_units
            or base_manifest.get("covered_unit_ids_sha256")
            != _stable_sha256(actual_units)
            or relevance_manifest.get("selected_unit_ids") != expected_units
            or relevance_manifest.get("selected_unit_ids_sha256")
            != _stable_sha256(expected_units)
        ):
            raise OpenRouterError(
                "Context-join base has incomplete or duplicate relevant-fact coverage"
            )
        for task in base_tasks:
            payload = task.get("payload")
            binding = (
                payload.get("critic_context_binding")
                if isinstance(payload, dict)
                else None
            )
            facts = (
                payload.get("shared_context_facts")
                if isinstance(payload, dict)
                else None
            )
            if not isinstance(binding, dict) or not isinstance(facts, list):
                raise OpenRouterError("Context-join task payload is corrupt")
            for fact_unit in facts:
                if (
                    not isinstance(fact_unit, dict)
                    or fact_unit.get("unit_content_sha256")
                    != _context_fact_unit_content_sha256(fact_unit)
                ):
                    raise OpenRouterError(
                        "Context-join exact fact content was mutated"
                    )
            binding_core = {
                key: value
                for key, value in binding.items()
                if key != "binding_sha256"
            }
            if (
                binding.get("binding_sha256") != _stable_sha256(binding_core)
                or task.get("payload_sha256") != _stable_sha256(payload)
                or binding.get("base_leaf_id")
                != base_manifest.get("base_leaf_id")
                or task.get("base_leaf_id")
                != base_manifest.get("base_leaf_id")
                or binding.get("base_payload_sha256")
                != base_manifest.get("base_payload_sha256")
                or task.get("base_kind") != base_manifest.get("base_kind")
                or task.get("base_index") != base_manifest.get("base_index")
                or binding.get("assigned_answer_ids")
                != base_manifest.get("assigned_answer_ids")
                or binding.get("answer_query_bindings")
                != base_manifest.get("answer_query_bindings")
                or task.get("answer_query_bindings")
                != base_manifest.get("answer_query_bindings")
                or binding.get("assigned_context_unit_ids")
                != task.get("context_unit_ids")
                or binding.get("assigned_context_unit_count") != len(facts)
                or binding.get("assigned_context_units_sha256")
                != _stable_sha256(facts)
                or binding.get("source_sha256")
                != fact_manifest.get("source_sha256")
                or binding.get("facts_sha256")
                != fact_manifest.get("facts_sha256")
                or binding.get("fact_count") != fact_manifest.get("fact_count")
                or binding.get("complete_unit_count")
                != fact_manifest.get("unit_count")
                or binding.get("complete_unit_ids_sha256")
                != fact_manifest.get("unit_ids_sha256")
                or binding.get("context_shard_index")
                != task.get("context_shard_index")
                or binding.get("context_shard_count")
                != task.get("context_shard_count")
            ):
                raise OpenRouterError(
                    "Context-join task binding or content digest was mutated"
                )
            _validate_context_join_physical_preflight(task)
        covered_tasks.extend(base_task_ids)
    if covered_tasks != actual_task_ids:
        raise OpenRouterError(
            "Context-join base manifests do not cover every task exactly once"
        )
    if results is None:
        return
    result_ids = [str(result.get("task_id") or "") for result in results]
    if result_ids != actual_task_ids or len(result_ids) != len(set(result_ids)):
        raise OpenRouterError(
            "Context-join results have missing, duplicate, or reordered lineage"
        )
    for task, result in zip(tasks, results, strict=True):
        if (
            result.get("base_leaf_id") != task.get("base_leaf_id")
            or result.get("assigned_answer_ids")
            != task.get("assigned_answer_ids")
            or result.get("context_unit_ids")
            != task.get("context_unit_ids")
            or result.get("input_sha256") != task.get("payload_sha256")
            or not isinstance(result.get("review"), dict)
        ):
            raise OpenRouterError(
                "Context-join result is cross-bound or tampered"
            )


def _aggregate_context_join_results(
    *,
    tasks: list[dict[str, Any]],
    results: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    """Collapse exact joint shards into one code-owned result per base leaf."""

    _validate_context_join_coverage(tasks, manifest, results)
    by_task_id = {str(result["task_id"]): result for result in results}
    aggregated: list[dict[str, Any]] = []
    for base_manifest in manifest["base_manifests"]:
        base_results = [
            by_task_id[str(task_id)] for task_id in base_manifest["task_ids"]
        ]
        merged = _merge_reviews_preserving_material_findings(
            [result["review"] for result in base_results],
            _critic_review_pass_seed(
                "Все точные context shards для answer/query leaf учтены."
            ),
        )
        merged = _collapse_answer_overlap_duplicates(merged)
        usage: dict[str, Any] = {}
        for key in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "reasoning_tokens",
            "cost",
        ):
            values = [
                result.get("usage", {}).get(key)
                for result in base_results
                if isinstance(result.get("usage"), dict)
                and isinstance(result["usage"].get(key), (int, float))
                and not isinstance(result["usage"].get(key), bool)
            ]
            if values:
                usage[key] = sum(values)
        usage["_aiv_context_join"] = {
            "version": CRITIC_MAP_REDUCE_VERSION,
            "base_leaf_id": base_manifest["base_leaf_id"],
            "task_ids": list(base_manifest["task_ids"]),
            "task_count": len(base_results),
            "exact_unit_accounting": True,
        }
        first = base_results[0]
        raw_text = json.dumps(
            {
                "version": CRITIC_MAP_REDUCE_VERSION,
                "mode": "deterministic_context_join_union",
                "base_leaf_id": base_manifest["base_leaf_id"],
                "task_receipts": [
                    {
                        "task_id": result["task_id"],
                        "context_unit_ids": result["context_unit_ids"],
                        "review_sha256": _stable_sha256(result["review"]),
                        "raw_response_sha256": hashlib.sha256(
                            str(result["raw_text"]).encode("utf-8")
                        ).hexdigest(),
                    }
                    for result in base_results
                ],
                "review_sha256": _stable_sha256(merged),
            },
            ensure_ascii=False,
        )
        aggregated.append(
            {
                "leaf_index": int(base_manifest["base_index"]),
                "answer_id": first.get("answer_id"),
                "assigned_answer_ids": list(
                    base_manifest["assigned_answer_ids"]
                ),
                "unit_id": first.get("unit_id"),
                "unit_index": first.get("unit_index"),
                "review": merged,
                "raw_text": raw_text,
                "usage": _critic_usage(usage),
                "input": {
                    "version": CRITIC_MAP_REDUCE_VERSION,
                    "mode": "context_join_receipts",
                    "base_leaf_id": base_manifest["base_leaf_id"],
                    "task_ids": list(base_manifest["task_ids"]),
                    "coverage_sha256": base_manifest[
                        "covered_unit_ids_sha256"
                    ],
                },
                "kind": str(base_manifest["base_kind"]),
            }
        )
    return aggregated


async def _reduce_shared_context_reviews(
    *,
    shared_results: list[dict[str, Any]],
    partition_manifest: dict[str, Any],
    semantic_inventory: dict[str, Any],
    iteration: int,
    recovery_final: bool,
    input_budget_bytes: int,
    context_envelope: dict[str, Any] | None = None,
    audit_sink: CriticCallAuditSink | None = None,
    transport_audit_checkpoint: AuditCheckpoint | None = None,
    transport_resume_lookup: TransportResumeLookup | None = None,
) -> dict[str, Any]:
    expected_unit_ids = [
        str(result["unit_id"])
        for result in sorted(
            shared_results,
            key=lambda value: int(value["unit_index"]),
        )
    ]
    if (
        not expected_unit_ids
        or len(expected_unit_ids) != len(set(expected_unit_ids))
    ):
        raise OpenRouterError(
            "Critic shared-context reducer requires unique lossless units"
        )

    def build_payload(
        nodes: list[dict[str, Any]],
        level: int,
        group_index: int,
    ) -> dict[str, Any]:
        lineage = [
            str(value) for node in nodes for value in node["lineage"]
        ]
        return {
            "version": CRITIC_MAP_REDUCE_VERSION,
            "stage": "shared_context_reduce",
            "iteration": iteration,
            "recovery_final": recovery_final,
            "shared_context_manifest": partition_manifest.get(
                "shared_context"
            ),
            "reduction_tree": {
                "level": level,
                "group_index": group_index,
                "lineage": (
                    lineage
                    if level == 0
                    else _review_reduce_lineage_descriptor(lineage)
                ),
                "expected_source_count": len(expected_unit_ids),
            },
            "context_reviews": [
                {
                    "unit_lineage": _review_reduce_lineage_descriptor(
                        list(node["lineage"])
                    ),
                    "review_sha256": _stable_sha256(node["review"]),
                    "review": node["review"],
                }
                for node in nodes
            ],
            "verdict_floor": _merge_reviews_preserving_material_findings(
                [node["review"] for node in nodes],
                _critic_review_pass_seed(
                    "Кодовый verdict floor общего контекста."
                ),
            )["verdict"],
        }

    reduced = await _hierarchical_review_reduce(
        source_nodes=[
            {
                "lineage": [str(result["unit_id"])],
                "review": result["review"],
            }
            for result in sorted(
                shared_results,
                key=lambda value: int(value["unit_index"]),
            )
        ],
        expected_lineage=expected_unit_ids,
        build_payload=build_payload,
        system_prompt=CRITIC_SHARED_CONTEXT_REDUCE_SYSTEM,
        schema_name_prefix=(
            f"aiv_analysis_critic_{iteration}_shared_context_reducer"
        ),
        owner="Critic shared-context reducer",
        audit_kind="shared_context_reducer",
        iteration=iteration,
        recovery_final=recovery_final,
        input_budget_bytes=input_budget_bytes,
        context_envelope=context_envelope,
        audit_sink=audit_sink,
        transport_audit_checkpoint=transport_audit_checkpoint,
        transport_resume_lookup=transport_resume_lookup,
    )
    return {
        **reduced,
        "digest": _shared_context_answer_digest(
            review=reduced["review"],
            partition_manifest=partition_manifest,
            semantic_inventory=semantic_inventory,
        ),
    }


async def _reduce_fragmented_answer(
    payload: dict[str, Any],
    *,
    answer_id: int,
    fragment_results: list[dict[str, Any]],
    fragment_manifest: dict[str, Any],
    iteration: int,
    recovery_final: bool,
    input_budget_bytes: int,
    context_envelope: dict[str, Any] | None = None,
    audit_sink: CriticCallAuditSink | None = None,
    shared_context_digest: dict[str, Any] | None = None,
    transport_audit_checkpoint: AuditCheckpoint | None = None,
    transport_resume_lookup: TransportResumeLookup | None = None,
) -> dict[str, Any]:
    """Reduce every core unit of one answer before corpus-level reduction."""

    expected_unit_ids = [
        str(value)
        for value in fragment_manifest.get("core_unit_ids") or []
    ]
    actual_unit_ids = [
        str(result.get("unit_id") or "") for result in fragment_results
    ]
    if (
        actual_unit_ids != expected_unit_ids
        or len(actual_unit_ids) != len(set(actual_unit_ids))
    ):
        missing = sorted(set(expected_unit_ids) - set(actual_unit_ids))
        duplicate = sorted(
            unit_id
            for unit_id in set(actual_unit_ids)
            if actual_unit_ids.count(unit_id) > 1
        )
        unexpected = sorted(set(actual_unit_ids) - set(expected_unit_ids))
        raise OpenRouterError(
            "Critic fragmented-answer reducer refused incomplete lineage: "
            f"answer_id={answer_id}, missing={missing}, "
            f"duplicate={duplicate}, unexpected={unexpected}"
        )
    complete_answers = _ordered_complete_answers(payload)
    answer_by_id = {
        int(answer["answer_id"]): answer for answer in complete_answers
    }
    if answer_id not in answer_by_id:
        raise OpenRouterError(
            f"Critic fragmented-answer reducer has no answer_id={answer_id}"
        )
    shared = (
        {"shared_context_digest": shared_context_digest}
        if shared_context_digest is not None
        else {
            key: value
            for key, value in payload.items()
            if key != "answers"
        }
    )
    result_by_unit = {
        str(result["unit_id"]): result for result in fragment_results
    }

    def build_payload(
        nodes: list[dict[str, Any]],
        level: int,
        group_index: int,
    ) -> dict[str, Any]:
        lineage = [
            str(value) for node in nodes for value in node["lineage"]
        ]
        lineage_descriptor = _review_reduce_lineage_descriptor(lineage)
        return {
            "version": CRITIC_MAP_REDUCE_VERSION,
            "stage": "fragmented_answer_reduce",
            "iteration": iteration,
            "recovery_final": recovery_final,
            "audit_payload_sha256": _stable_sha256(payload),
            **shared,
            "answer_id": answer_id,
            "complete_answer_index": [
                _critic_answer_index(answer_by_id[answer_id])
            ],
            "complete_answer_manifests": [
                _critic_complete_answer_manifest(answer_by_id[answer_id])
            ],
            "fragment_manifest": {
                "manifest_sha256": _stable_sha256(fragment_manifest),
                "core_unit_count": len(expected_unit_ids),
                "group_core_unit_ids": lineage if level == 0 else [],
                "group_lineage": lineage_descriptor,
                "exact_core_accounting": bool(
                    fragment_manifest.get("exact_core_accounting")
                ),
            },
            "reduction_tree": {
                "level": level,
                "group_index": group_index,
                "lineage": lineage if level == 0 else lineage_descriptor,
                "expected_source_count": len(expected_unit_ids),
            },
            "fragment_reviews": [
                {
                    "unit_ids": (
                        list(node["lineage"]) if level == 0 else []
                    ),
                    "unit_lineage": _review_reduce_lineage_descriptor(
                        list(node["lineage"])
                    ),
                    "unit_indexes": [
                        int(result_by_unit[str(unit_id)]["unit_index"])
                        for unit_id in node["lineage"]
                        if str(unit_id) in result_by_unit
                    ] if level == 0 else [],
                    "review_sha256": _stable_sha256(node["review"]),
                    "review": node["review"],
                }
                for node in nodes
            ],
            "verdict_floor": _merge_reviews_preserving_material_findings(
                [node["review"] for node in nodes],
                _critic_review_pass_seed(
                    "Кодовый verdict floor для группы фрагментов."
                ),
            )["verdict"],
        }

    reduced = await _hierarchical_review_reduce(
        source_nodes=[
            {
                "lineage": [str(result["unit_id"])],
                "review": result["review"],
            }
            for result in fragment_results
        ],
        expected_lineage=expected_unit_ids,
        build_payload=build_payload,
        system_prompt=CRITIC_ANSWER_REDUCE_SYSTEM,
        schema_name_prefix=(
            f"aiv_analysis_critic_{iteration}_answer_{answer_id}_reducer"
        ),
        owner=f"Critic answer {answer_id} reducer",
        audit_kind="answer_reducer",
        iteration=iteration,
        recovery_final=recovery_final,
        input_budget_bytes=input_budget_bytes,
        context_envelope=context_envelope,
        audit_sink=audit_sink,
        transport_audit_checkpoint=transport_audit_checkpoint,
        transport_resume_lookup=transport_resume_lookup,
        collapse_answer_duplicates=True,
    )
    provider_calls = list(reduced["provider_calls"])
    raw_text = json.dumps(
        {
            "version": CRITIC_MAP_REDUCE_VERSION,
            "mode": "hierarchical_answer_reduce",
            "answer_id": answer_id,
            "tree_levels": reduced["tree_levels"],
            "provider_calls": provider_calls,
            "final_review_sha256": _stable_sha256(reduced["review"]),
        },
        ensure_ascii=False,
    )
    return {
        "answer_id": answer_id,
        "assigned_answer_ids": [answer_id],
        "review": reduced["review"],
        "raw_text": raw_text,
        "usage": _aggregate_provider_call_usage(provider_calls),
        "input": {
            "version": CRITIC_MAP_REDUCE_VERSION,
            "mode": "hierarchical_answer_reduce",
            "answer_id": answer_id,
            "tree_levels": reduced["tree_levels"],
        },
        "status": reduced["status"],
        "request_utf8_bytes": reduced["max_request_utf8_bytes"],
        "provider_calls": provider_calls,
    }


async def _reduce_corpus_reviews(
    payload: dict[str, Any],
    *,
    corpus_results: list[dict[str, Any]],
    partition_manifest: dict[str, Any],
    iteration: int,
    recovery_final: bool,
    input_budget_bytes: int,
    context_envelope: dict[str, Any] | None = None,
    audit_sink: CriticCallAuditSink | None = None,
    shared_context_digest: dict[str, Any] | None = None,
    transport_audit_checkpoint: AuditCheckpoint | None = None,
    transport_resume_lookup: TransportResumeLookup | None = None,
) -> dict[str, Any]:
    """Reduce all answer-level reviews through a bounded fan-in tree."""

    expected_answer_ids = [
        int(value) for value in partition_manifest["complete_answer_ids"]
    ]
    actual_answer_ids = [
        int(value)
        for result in corpus_results
        for value in result["assigned_answer_ids"]
    ]
    if (
        actual_answer_ids != expected_answer_ids
        or len(actual_answer_ids) != len(set(actual_answer_ids))
    ):
        raise OpenRouterError(
            "Critic corpus reducer refused incomplete answer lineage: "
            f"expected={expected_answer_ids}, actual={actual_answer_ids}"
        )
    complete_answers = _ordered_complete_answers(payload)
    answer_by_id = {
        int(answer["answer_id"]): answer for answer in complete_answers
    }
    shared = (
        {"shared_context_digest": shared_context_digest}
        if shared_context_digest is not None
        else {
            key: value for key, value in payload.items() if key != "answers"
        }
    )

    def build_payload(
        nodes: list[dict[str, Any]],
        level: int,
        group_index: int,
    ) -> dict[str, Any]:
        lineage = [
            int(value) for node in nodes for value in node["lineage"]
        ]
        lineage_descriptor = _review_reduce_lineage_descriptor(lineage)
        return {
            "version": CRITIC_MAP_REDUCE_VERSION,
            "stage": "corpus_reduce",
            "iteration": iteration,
            "recovery_final": recovery_final,
            "audit_payload_sha256": _stable_sha256(payload),
            **shared,
            "complete_answer_index": [
                _critic_answer_index(answer_by_id[answer_id])
                for answer_id in lineage
            ] if level == 0 else [],
            "complete_answer_manifests": [
                _critic_complete_answer_manifest(answer_by_id[answer_id])
                for answer_id in lineage
            ] if level == 0 else [],
            "partition_manifest": {
                "manifest_sha256": _stable_sha256(partition_manifest),
                "complete_answer_count": len(expected_answer_ids),
                "group_answer_ids": lineage if level == 0 else [],
                "group_lineage": lineage_descriptor,
                "exact_accounting": bool(
                    partition_manifest.get("exact_accounting")
                ),
            },
            "reduction_tree": {
                "level": level,
                "group_index": group_index,
                "lineage": lineage if level == 0 else lineage_descriptor,
                "expected_source_count": len(expected_answer_ids),
            },
            "leaf_reviews": [
                {
                    "assigned_answer_ids": (
                        list(node["lineage"]) if level == 0 else []
                    ),
                    "answer_lineage": _review_reduce_lineage_descriptor(
                        list(node["lineage"])
                    ),
                    "review_sha256": _stable_sha256(node["review"]),
                    "review": node["review"],
                }
                for node in nodes
            ],
            "verdict_floor": _merge_reviews_preserving_material_findings(
                [node["review"] for node in nodes],
                _critic_review_pass_seed(
                    "Кодовый verdict floor для группы ответов."
                ),
            )["verdict"],
        }

    reduced = await _hierarchical_review_reduce(
        source_nodes=[
            {
                "lineage": [
                    int(value) for value in result["assigned_answer_ids"]
                ],
                "review": result["review"],
            }
            for result in corpus_results
        ],
        expected_lineage=expected_answer_ids,
        build_payload=build_payload,
        system_prompt=CRITIC_REDUCE_SYSTEM,
        schema_name_prefix=f"aiv_analysis_critic_{iteration}_corpus_reducer",
        owner="Critic corpus reducer",
        audit_kind="corpus_reducer",
        iteration=iteration,
        recovery_final=recovery_final,
        input_budget_bytes=input_budget_bytes,
        context_envelope=context_envelope,
        audit_sink=audit_sink,
        transport_audit_checkpoint=transport_audit_checkpoint,
        transport_resume_lookup=transport_resume_lookup,
    )
    provider_calls = list(reduced["provider_calls"])
    raw_text = json.dumps(
        {
            "version": CRITIC_MAP_REDUCE_VERSION,
            "mode": "hierarchical_corpus_reduce",
            "tree_levels": reduced["tree_levels"],
            "provider_calls": provider_calls,
            "final_review_sha256": _stable_sha256(reduced["review"]),
        },
        ensure_ascii=False,
    )
    return {
        "review": reduced["review"],
        "raw_text": raw_text,
        "usage": _aggregate_provider_call_usage(provider_calls),
        "input": {
            "version": CRITIC_MAP_REDUCE_VERSION,
            "mode": "hierarchical_corpus_reduce",
            "tree_levels": reduced["tree_levels"],
        },
        "status": reduced["status"],
        "request_utf8_bytes": reduced["max_request_utf8_bytes"],
        "provider_calls": provider_calls,
    }


def _aggregate_map_reduce_usage(
    *,
    call_records: list[dict[str, Any]],
    partition_manifest: dict[str, Any],
    context_envelope: dict[str, Any],
    final_usage: dict[str, Any] | None,
    reducer_status: str,
) -> dict[str, Any]:
    output = dict(final_usage or {})
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "reasoning_tokens",
        "cost",
    ):
        values = [
            record.get("usage", {}).get(key)
            for record in call_records
            if isinstance(record.get("usage"), dict)
        ]
        numeric = [
            value
            for value in values
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        if numeric:
            output[key] = sum(numeric)
    output = _critic_usage(output)
    output["_aiv_critic_map_reduce"] = {
        **partition_manifest,
        "context_envelope": _json_safe(context_envelope),
        "reducer_status": reducer_status,
        "child_calls": call_records,
    }
    consumed_repairs = [
        {
            "kind": record["kind"],
            "index": record["index"],
            "attempts": record["usage"].get("_aiv_critic_attempts"),
        }
        for record in call_records
        if isinstance(record.get("usage"), dict)
        and isinstance(record["usage"].get("_aiv_critic_attempts"), list)
        and record["usage"]["_aiv_critic_attempts"]
    ]
    if consumed_repairs:
        # The analyzer uses this key to forbid a second semantic rewrite.
        output["_aiv_critic_attempts"] = consumed_repairs
    return _json_safe(output)


async def _review_analysis_map_reduce(
    payload: dict[str, Any],
    *,
    iteration: int,
    max_iterations: int,
    recovery_final: bool,
    input_budget_bytes: int,
    context_envelope: dict[str, Any],
    audit_sink: CriticCallAuditSink | None = None,
    transport_audit_checkpoint: AuditCheckpoint | None = None,
    transport_resume_lookup: TransportResumeLookup | None = None,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    whole_leaves, fragment_tasks, partition_manifest = _build_critic_map_plan(
        payload,
        iteration=iteration,
        max_iterations=max_iterations,
        recovery_final=recovery_final,
        input_budget_bytes=input_budget_bytes,
        context_envelope=context_envelope,
    )
    shared_context_tasks = [
        task
        for task in fragment_tasks
        if task.get("kind") == "shared_context"
    ]
    answer_fragment_tasks = [
        task
        for task in fragment_tasks
        if task.get("kind") != "shared_context"
    ]
    if partition_manifest["leaf_count"] < 2:
        raise OpenRouterError(
            "Critic payload exceeded the direct window but could not be split "
            "into multiple complete evidence leaves"
        )

    semaphore = asyncio.Semaphore(CRITIC_MAP_CONCURRENCY)

    async def audited_leaf_call(
        *,
        kind: str,
        index: int,
        leaf_payload: dict[str, Any],
        lineage: dict[str, Any],
        schema_name: str,
        invoke: Callable[
            [],
            Awaitable[tuple[dict[str, Any], str, dict[str, Any]]],
        ],
    ) -> tuple[dict[str, Any], str, dict[str, Any]]:
        """Persist a paid leaf result before any sibling or reducer can fail."""

        descriptor = _critic_logical_call_descriptor(
            iteration=iteration,
            kind=kind,
            index=index,
            schema_name=schema_name,
            call_input=leaf_payload,
            lineage=lineage,
        )
        attempt_id = _critic_attempt_id(descriptor)
        cached = await _lookup_completed_critic_call(audit_sink, descriptor)
        if cached is not None:
            return cached

        async def emit(
            *,
            status: str,
            review: dict[str, Any] | None,
            raw_text: str,
            usage: dict[str, Any],
            provider_response_present: bool,
            error: BaseException | None,
        ) -> None:
            await _emit_critic_call_audit(
                audit_sink,
                attempt_id=attempt_id,
                iteration=iteration,
                kind=kind,
                index=index,
                call_input=leaf_payload,
                lineage=lineage,
                status=status,
                review=review,
                raw_text=raw_text,
                usage=usage,
                provider_response_present=provider_response_present,
                error=error,
                schema_name=schema_name,
            )

        try:
            review, raw_text, usage = await invoke()
        except BaseException as exc:
            result = getattr(exc, "result", None)
            await emit(
                status=(
                    "cancelled"
                    if isinstance(exc, asyncio.CancelledError)
                    else "failed"
                ),
                review=None,
                raw_text=str(getattr(result, "text", "") or ""),
                usage=(
                    dict(getattr(result, "usage", {}) or {})
                    if result is not None
                    else {}
                ),
                provider_response_present=result is not None,
                error=exc,
            )
            raise
        await emit(
            status="completed",
            review=review,
            raw_text=raw_text,
            usage=usage,
            provider_response_present=True,
            error=None,
        )
        return review, raw_text, usage

    async def run_whole_leaf(
        leaf_index: int,
        leaf_payload: dict[str, Any],
    ) -> dict[str, Any]:
        async def invoke() -> tuple[dict[str, Any], str, dict[str, Any]]:
            async with semaphore:
                return await _review_analysis_once(
                    leaf_payload,
                    iteration=iteration,
                    max_iterations=max_iterations,
                    recovery_final=recovery_final,
                    schema_name=(
                        f"aiv_analysis_critic_{iteration}_leaf_{leaf_index}"
                    ),
                    map_leaf=True,
                    validate_schema=True,
                    audit_sink=audit_sink,
                    transport_audit_checkpoint=transport_audit_checkpoint,
                    transport_resume_lookup=transport_resume_lookup,
                )

        assigned_answer_ids = list(
            leaf_payload["critic_map_partition"]["assigned_answer_ids"]
        )
        schema_name = f"aiv_analysis_critic_{iteration}_leaf_{leaf_index}"
        review, raw_text, usage = await audited_leaf_call(
            kind="leaf",
            index=leaf_index,
            leaf_payload=leaf_payload,
            lineage={"assigned_answer_ids": assigned_answer_ids},
            schema_name=schema_name,
            invoke=invoke,
        )
        return {
            "leaf_index": leaf_index,
            "assigned_answer_ids": assigned_answer_ids,
            "review": review,
            "raw_text": raw_text,
            "usage": usage,
            "input": leaf_payload,
            "kind": "leaf",
        }

    async def run_fragment_leaf(task: dict[str, Any]) -> dict[str, Any]:
        leaf_payload = task["payload"]
        unit_id = str(task["unit_id"])
        schema_name = (
            "aiv_analysis_critic_"
            f"{iteration}_answer_{task['answer_id']}_fragment_"
            f"{task['unit_index']}"
        )
        async def invoke() -> tuple[dict[str, Any], str, dict[str, Any]]:
            async with semaphore:
                return await _review_analysis_once(
                    leaf_payload,
                    iteration=iteration,
                    max_iterations=max_iterations,
                    recovery_final=recovery_final,
                    schema_name=(
                        "aiv_analysis_critic_"
                        f"{iteration}_answer_{task['answer_id']}_fragment_"
                        f"{task['unit_index']}"
                    ),
                    map_leaf=True,
                    fragment_leaf=True,
                    validate_schema=True,
                    audit_sink=audit_sink,
                    transport_audit_checkpoint=transport_audit_checkpoint,
                    transport_resume_lookup=transport_resume_lookup,
                )

        review, raw_text, usage = await audited_leaf_call(
            kind="fragment_leaf",
            index=int(task["unit_index"]),
            leaf_payload=leaf_payload,
            lineage={
                "answer_id": int(task["answer_id"]),
                "unit_id": unit_id,
                "unit_index": int(task["unit_index"]),
            },
            schema_name=schema_name,
            invoke=invoke,
        )
        return {
            "leaf_index": int(task["unit_index"]),
            "answer_id": int(task["answer_id"]),
            "assigned_answer_ids": [int(task["answer_id"])],
            "unit_id": unit_id,
            "unit_index": int(task["unit_index"]),
            "review": review,
            "raw_text": raw_text,
            "usage": usage,
            "input": leaf_payload,
            "kind": "fragment_leaf",
        }

    async def run_shared_context_leaf(task: dict[str, Any]) -> dict[str, Any]:
        leaf_payload = task["payload"]
        unit_id = str(task["unit_id"])
        unit_index = int(task["unit_index"])
        schema_name = (
            f"aiv_analysis_critic_{iteration}_shared_context_{unit_index}"
        )

        async def invoke() -> tuple[dict[str, Any], str, dict[str, Any]]:
            async with semaphore:
                return await _review_analysis_once(
                    leaf_payload,
                    iteration=iteration,
                    max_iterations=max_iterations,
                    recovery_final=recovery_final,
                    schema_name=schema_name,
                    shared_context_leaf=True,
                    validate_schema=True,
                    audit_sink=audit_sink,
                    transport_audit_checkpoint=transport_audit_checkpoint,
                    transport_resume_lookup=transport_resume_lookup,
                )

        review, raw_text, usage = await audited_leaf_call(
            kind="shared_context_leaf",
            index=unit_index,
            leaf_payload=leaf_payload,
            lineage={
                "unit_id": unit_id,
                "unit_index": unit_index,
                "source_sha256": leaf_payload[
                    "critic_shared_context_partition"
                ]["source_sha256"],
            },
            schema_name=schema_name,
            invoke=invoke,
        )
        return {
            "unit_id": unit_id,
            "unit_index": unit_index,
            "review": review,
            "raw_text": raw_text,
            "usage": usage,
            "input": leaf_payload,
            "kind": "shared_context_leaf",
        }

    async def run_context_join_leaf(task: dict[str, Any]) -> dict[str, Any]:
        leaf_payload = task["payload"]
        global_index = int(task["global_index"])
        schema_name = str(task["schema_name"])
        base_kind = str(task["base_kind"])

        async def invoke() -> tuple[dict[str, Any], str, dict[str, Any]]:
            async with semaphore:
                return await _review_analysis_once(
                    leaf_payload,
                    iteration=iteration,
                    max_iterations=max_iterations,
                    recovery_final=recovery_final,
                    schema_name=schema_name,
                    map_leaf=True,
                    fragment_leaf=base_kind == "fragment_leaf",
                    context_join_leaf=True,
                    validate_schema=True,
                    audit_sink=audit_sink,
                    transport_audit_checkpoint=transport_audit_checkpoint,
                    transport_resume_lookup=transport_resume_lookup,
                )

        review, raw_text, usage = await audited_leaf_call(
            kind="context_join_leaf",
            index=global_index,
            leaf_payload=leaf_payload,
            lineage={
                "task_id": task["task_id"],
                "base_leaf_id": task["base_leaf_id"],
                "assigned_answer_ids": task["assigned_answer_ids"],
                "context_unit_ids": task["context_unit_ids"],
                "physical_request_sha256": task[
                    "physical_request_sha256"
                ],
            },
            schema_name=schema_name,
            invoke=invoke,
        )
        return {
            "task_id": task["task_id"],
            "base_leaf_id": task["base_leaf_id"],
            "base_kind": base_kind,
            "base_index": int(task["base_index"]),
            "answer_id": (
                int(task["assigned_answer_ids"][0])
                if base_kind == "fragment_leaf"
                else None
            ),
            "assigned_answer_ids": list(task["assigned_answer_ids"]),
            "unit_id": (
                str(
                    leaf_payload.get("answers", [{}])[0]
                    .get("critic_fragment", {})
                    .get("unit_id")
                    or ""
                )
                if base_kind == "fragment_leaf"
                else None
            ),
            "unit_index": (
                int(
                    leaf_payload.get("answers", [{}])[0]
                    .get("critic_fragment", {})
                    .get("unit_index")
                    or 0
                )
                if base_kind == "fragment_leaf"
                else None
            ),
            "context_unit_ids": list(task["context_unit_ids"]),
            "review": review,
            "raw_text": raw_text,
            "usage": usage,
            "input": leaf_payload,
            "input_sha256": _stable_sha256(leaf_payload),
            "kind": "context_join_leaf",
        }

    shared_context_results: list[dict[str, Any]] = []
    shared_context_reduced: dict[str, Any] | None = None
    shared_context_digest: dict[str, Any] | None = None
    context_join_tasks: list[dict[str, Any]] = []
    context_join_results: list[dict[str, Any]] = []
    context_join_manifest: dict[str, Any] | None = None
    if shared_context_tasks:
        context_tasks = [
            asyncio.create_task(
                run_shared_context_leaf(task),
                name=(
                    f"aiv-critic-shared-context-{iteration}-"
                    f"{task['unit_index']}"
                ),
            )
            for task in shared_context_tasks
        ]
        try:
            context_outcomes = await asyncio.gather(
                *context_tasks,
                return_exceptions=True,
            )
        except asyncio.CancelledError:
            for task in context_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*context_tasks, return_exceptions=True)
            raise
        context_failures = [
            outcome
            for outcome in context_outcomes
            if isinstance(outcome, BaseException)
        ]
        if context_failures:
            first_failure = next(
                (
                    outcome
                    for outcome in context_failures
                    if not isinstance(outcome, asyncio.CancelledError)
                ),
                context_failures[0],
            )
            raise OpenRouterError(
                "Critic shared-context leaves did not all complete after "
                f"durable per-call audit: failures={len(context_failures)}"
            ) from first_failure
        shared_context_results = sorted(
            [
                outcome
                for outcome in context_outcomes
                if isinstance(outcome, dict)
            ],
            key=lambda result: int(result["unit_index"]),
        )
        semantic_inventory = _shared_context_semantic_inventory(payload)
        shared_context_reduced = await _reduce_shared_context_reviews(
            shared_results=shared_context_results,
            partition_manifest=partition_manifest,
            semantic_inventory=semantic_inventory,
            iteration=iteration,
            recovery_final=recovery_final,
            input_budget_bytes=input_budget_bytes,
            context_envelope=context_envelope,
            audit_sink=audit_sink,
            transport_audit_checkpoint=transport_audit_checkpoint,
            transport_resume_lookup=transport_resume_lookup,
        )
        shared_context_digest = shared_context_reduced["digest"]
        base_entries = [
            {
                "base_kind": "leaf",
                "base_index": index,
                "payload": leaf_payload,
            }
            for index, leaf_payload in enumerate(whole_leaves)
        ]
        base_entries.extend(
            {
                "base_kind": "fragment_leaf",
                "base_index": len(whole_leaves) + index,
                "payload": task["payload"],
            }
            for index, task in enumerate(answer_fragment_tasks)
        )
        reserve = int(
            partition_manifest.get("shared_context", {}).get(
                "digest_reserve_bytes"
            )
            or max(4_096, input_budget_bytes // 3)
        )
        context_join_tasks, context_join_manifest = (
            _build_context_join_tasks(
                base_entries=base_entries,
                semantic_inventory=semantic_inventory,
                context_receipt=shared_context_digest,
                iteration=iteration,
                max_iterations=max_iterations,
                recovery_final=recovery_final,
                input_budget_bytes=input_budget_bytes,
                context_envelope=context_envelope,
                per_call_reserve_bytes=reserve,
            )
        )
        partition_manifest["context_join"] = {
            "manifest_sha256": _stable_sha256(context_join_manifest),
            "fact_manifest_sha256": context_join_manifest[
                "fact_manifest_sha256"
            ],
            "base_count": context_join_manifest["base_count"],
            "task_count": context_join_manifest["task_count"],
            "task_ids_sha256": context_join_manifest["task_ids_sha256"],
            "mode": context_join_manifest["mode"],
            "exact_global_fact_accounting": True,
            "attached_relevant_unit_count": context_join_manifest[
                "attached_relevant_unit_count"
            ],
            "avoided_cross_product_unit_attachments": context_join_manifest[
                "avoided_cross_product_unit_attachments"
            ],
            "paid_task_cost_bound": context_join_manifest[
                "paid_task_cost_bound"
            ],
        }
        partition_manifest["leaf_count"] = (
            len(shared_context_tasks) + len(context_join_tasks)
        )
        partition_manifest["context_join_leaf_count"] = len(
            context_join_tasks
        )
        partition_manifest["leaf_request_utf8_bytes"] = [
            *(
                partition_manifest.get("shared_context", {}).get(
                    "shared_context_request_utf8_bytes", []
                )
            ),
            *[
                int(task["physical_request_utf8_bytes"])
                for task in context_join_tasks
            ],
        ]
        partition_manifest["shared_context_reducer_status"] = (
            shared_context_reduced["status"]
        )

    if context_join_tasks:
        map_tasks = [
            asyncio.create_task(
                run_context_join_leaf(task),
                name=(
                    f"aiv-critic-context-join-{iteration}-"
                    f"{task['global_index']}"
                ),
            )
            for task in context_join_tasks
        ]
    else:
        map_tasks = [
            asyncio.create_task(
                run_whole_leaf(index, leaf),
                name=f"aiv-critic-leaf-{iteration}-{index}",
            )
            for index, leaf in enumerate(whole_leaves)
        ]
        map_tasks.extend(
            asyncio.create_task(
                run_fragment_leaf(task),
                name=(
                    f"aiv-critic-fragment-{iteration}-"
                    f"{task['answer_id']}-{task['unit_index']}"
                ),
            )
            for task in answer_fragment_tasks
        )
    try:
        map_outcomes = await asyncio.gather(
            *map_tasks,
            return_exceptions=True,
        )
    except asyncio.CancelledError:
        # Outer cancellation is different from a sibling failure: stop any
        # outstanding requests, then wait for every wrapper to append its
        # explicit cancelled/failed audit record before propagating.
        for task in map_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*map_tasks, return_exceptions=True)
        raise

    failures = [
        outcome
        for outcome in map_outcomes
        if isinstance(outcome, BaseException)
    ]
    if failures:
        cancelled_count = sum(
            isinstance(outcome, asyncio.CancelledError)
            for outcome in failures
        )
        failed_count = len(failures) - cancelled_count
        first_failure = next(
            (
                outcome
                for outcome in failures
                if not isinstance(outcome, asyncio.CancelledError)
            ),
            failures[0],
        )
        raise OpenRouterError(
            "Critic map leaves did not all complete after durable per-call "
            f"audit: failed={failed_count}, cancelled={cancelled_count}"
        ) from first_failure
    mapped_results = [
        outcome
        for outcome in map_outcomes
        if isinstance(outcome, dict)
    ]
    if context_join_tasks:
        if context_join_manifest is None or shared_context_digest is None:
            raise OpenRouterError(
                "Context-join execution has no bound manifest/receipt"
            )
        context_join_results = mapped_results
        aggregated_results = _aggregate_context_join_results(
            tasks=context_join_tasks,
            results=context_join_results,
            manifest=context_join_manifest,
        )
        shared_context_digest = {
            **shared_context_digest,
            "status": "completed_lossless_answer_context_join",
            "semantic_context": {
                **shared_context_digest["semantic_context"],
                "delivery": "complete_lossless_joint_shards",
                "fact_manifest_sha256": context_join_manifest[
                    "fact_manifest_sha256"
                ],
                "base_count": context_join_manifest["base_count"],
                "task_count": context_join_manifest["task_count"],
                "task_ids_sha256": context_join_manifest[
                    "task_ids_sha256"
                ],
                "exact_cross_product_accounting": True,
            },
        }
        partition_manifest["shared_context_digest_sha256"] = _stable_sha256(
            shared_context_digest
        )
        whole_results = sorted(
            [
                result
                for result in aggregated_results
                if result["kind"] == "leaf"
            ],
            key=lambda result: int(result["leaf_index"]),
        )
        fragment_results = sorted(
            [
                result
                for result in aggregated_results
                if result["kind"] == "fragment_leaf"
            ],
            key=lambda result: (
                int(result["answer_id"]),
                int(result["unit_index"]),
            ),
        )
    else:
        whole_results = sorted(
            [result for result in mapped_results if result["kind"] == "leaf"],
            key=lambda result: int(result["leaf_index"]),
        )
        fragment_results = sorted(
            [
                result
                for result in mapped_results
                if result["kind"] == "fragment_leaf"
            ],
            key=lambda result: (
                int(result["answer_id"]),
                int(result["unit_index"]),
            ),
        )
    fragment_manifest_by_answer = {
        int(manifest["answer_id"]): manifest
        for manifest in partition_manifest["fragmented_answers"]
    }
    grouped_fragments: dict[int, list[dict[str, Any]]] = {}
    for result in fragment_results:
        grouped_fragments.setdefault(int(result["answer_id"]), []).append(result)

    answer_reducer_tasks = [
        asyncio.create_task(
            _reduce_fragmented_answer(
                payload,
                answer_id=answer_id,
                fragment_results=grouped_fragments[answer_id],
                fragment_manifest=fragment_manifest_by_answer[answer_id],
                iteration=iteration,
                recovery_final=recovery_final,
                input_budget_bytes=input_budget_bytes,
                audit_sink=audit_sink,
                shared_context_digest=shared_context_digest,
                transport_audit_checkpoint=transport_audit_checkpoint,
                transport_resume_lookup=transport_resume_lookup,
            ),
            name=f"aiv-critic-answer-reducer-{iteration}-{answer_id}",
        )
        for answer_id in sorted(grouped_fragments)
    ]
    try:
        answer_reducer_outcomes = await asyncio.gather(
            *answer_reducer_tasks,
            return_exceptions=True,
        )
    except asyncio.CancelledError:
        for task in answer_reducer_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(
            *answer_reducer_tasks,
            return_exceptions=True,
        )
        raise
    answer_reducer_failures = [
        outcome
        for outcome in answer_reducer_outcomes
        if isinstance(outcome, BaseException)
    ]
    if answer_reducer_failures:
        first_failure = next(
            (
                outcome
                for outcome in answer_reducer_failures
                if not isinstance(outcome, asyncio.CancelledError)
            ),
            answer_reducer_failures[0],
        )
        if isinstance(first_failure, OpenRouterResponseContractError):
            raise first_failure
        raise OpenRouterError(
            "Critic answer reducers did not all complete after durable "
            f"per-call audit: failures={len(answer_reducer_failures)}"
        ) from first_failure
    answer_reducer_results = [
        outcome
        for outcome in answer_reducer_outcomes
        if isinstance(outcome, dict)
    ]
    corpus_results: list[dict[str, Any]] = [*whole_results]
    corpus_results.extend(
        {
            **result,
            "leaf_index": len(whole_results) + index,
            "kind": "answer_reducer",
        }
        for index, result in enumerate(answer_reducer_results)
    )
    corpus_results.sort(
        key=lambda result: min(
            int(answer_id) for answer_id in result["assigned_answer_ids"]
        )
    )
    for index, result in enumerate(corpus_results):
        result["leaf_index"] = index

    source_reviews = [
        *(
            [shared_context_reduced["review"]]
            if shared_context_reduced is not None
            else []
        ),
        *(result["review"] for result in corpus_results),
    ]
    corpus_reducer = await _reduce_corpus_reviews(
        payload,
        corpus_results=corpus_results,
        partition_manifest=partition_manifest,
        iteration=iteration,
        recovery_final=recovery_final,
        input_budget_bytes=input_budget_bytes,
        context_envelope=context_envelope,
        audit_sink=audit_sink,
        shared_context_digest=shared_context_digest,
        transport_audit_checkpoint=transport_audit_checkpoint,
        transport_resume_lookup=transport_resume_lookup,
    )
    reducer_payload = corpus_reducer["input"]
    reducer_request_bytes = int(corpus_reducer["request_utf8_bytes"])
    reducer_raw_text = str(corpus_reducer["raw_text"])
    reducer_usage = corpus_reducer["usage"]
    reducer_status = str(corpus_reducer["status"])
    final_review = _merge_reviews_preserving_material_findings(
        source_reviews,
        corpus_reducer["review"],
    )
    if context_join_results:
        call_records = [
            _critic_call_provenance(
                kind="context_join_leaf",
                index=index,
                input_value=result["input"],
                raw_text=result["raw_text"],
                usage=result["usage"],
                verdict=str(result["review"].get("verdict") or "block"),
                lineage={
                    "task_id": result["task_id"],
                    "base_leaf_id": result["base_leaf_id"],
                    "assigned_answer_ids": result["assigned_answer_ids"],
                    "context_unit_ids": result["context_unit_ids"],
                },
            )
            for index, result in enumerate(context_join_results)
        ]
    else:
        call_records = [
            _critic_call_provenance(
                kind=str(result["kind"]),
                index=(
                    int(result["unit_index"])
                    if result["kind"] == "fragment_leaf"
                    else int(result["leaf_index"])
                ),
                input_value=result["input"],
                raw_text=result["raw_text"],
                usage=result["usage"],
                verdict=str(result["review"].get("verdict") or "block"),
                lineage=(
                    {
                        "answer_id": result["answer_id"],
                        "unit_id": result["unit_id"],
                        "unit_index": result["unit_index"],
                    }
                    if result["kind"] == "fragment_leaf"
                    else {
                        "assigned_answer_ids": result[
                            "assigned_answer_ids"
                        ]
                    }
                ),
            )
            for result in [*whole_results, *fragment_results]
        ]
    context_call_records = [
        _critic_call_provenance(
            kind="shared_context_leaf",
            index=int(result["unit_index"]),
            input_value=result["input"],
            raw_text=result["raw_text"],
            usage=result["usage"],
            verdict=str(result["review"].get("verdict") or "block"),
            lineage={
                "unit_id": result["unit_id"],
                "unit_index": result["unit_index"],
                "source_sha256": partition_manifest.get(
                    "shared_context",
                    {},
                ).get("source_sha256"),
            },
        )
        for result in shared_context_results
    ]
    if shared_context_reduced is not None:
        context_call_records.extend(
            _critic_call_provenance(
                kind="shared_context_reducer",
                index=(
                    int(call["level"]) * 100_000
                    + int(call["group_index"])
                ),
                input_value=call["input"],
                raw_text=call["raw_text"],
                usage=call["usage"],
                verdict=str(
                    shared_context_reduced["review"].get("verdict")
                    or "block"
                ),
                lineage={
                    "core_unit_ids": call["lineage"],
                    "tree_level": call["level"],
                },
            )
            for call in shared_context_reduced["provider_calls"]
        )
    call_records = [*context_call_records, *call_records]
    call_records.extend(
        _critic_call_provenance(
            kind="answer_reducer",
            index=(
                int(call["level"]) * 100_000
                + int(call["group_index"])
            ),
            input_value=call["input"],
            raw_text=call["raw_text"],
            usage=call["usage"],
            verdict=str(result["review"].get("verdict") or "block"),
            lineage={
                "answer_id": result["answer_id"],
                "core_unit_ids": call["lineage"],
                "tree_level": call["level"],
            },
        )
        for result in answer_reducer_results
        for call in result["provider_calls"]
    )
    call_records.extend(
        _critic_call_provenance(
            # Preserve the historical provenance label while append-only
            # audit artifacts use the explicit ``corpus_reducer`` kind.
            kind="reducer",
            index=(
                int(call["level"]) * 100_000
                + int(call["group_index"])
            ),
            input_value=call["input"],
            raw_text=call["raw_text"],
            usage=call["usage"],
            verdict=str(final_review.get("verdict") or "block"),
            lineage={
                "complete_answer_ids": call["lineage"],
                "tree_level": call["level"],
            },
        )
        for call in corpus_reducer["provider_calls"]
    )
    raw_text = json.dumps(
        {
            "version": CRITIC_MAP_REDUCE_VERSION,
            "mode": "map_reduce",
            "audit_payload_sha256": _stable_sha256(payload),
            "partition_manifest": partition_manifest,
            "shared_context": (
                {
                    "leaf_responses": [
                        {
                            "unit_id": result["unit_id"],
                            "unit_index": result["unit_index"],
                            "raw_text": result["raw_text"],
                            "raw_response_sha256": hashlib.sha256(
                                result["raw_text"].encode("utf-8")
                            ).hexdigest(),
                            "review_sha256": _stable_sha256(
                                result["review"]
                            ),
                        }
                        for result in shared_context_results
                    ],
                    "reducer_status": shared_context_reduced["status"],
                    "review_sha256": _stable_sha256(
                        shared_context_reduced["review"]
                    ),
                    "digest_sha256": _stable_sha256(
                        shared_context_digest
                    ),
                    "context_join_manifest": context_join_manifest,
                    "context_join_responses": [
                        {
                            "task_id": result["task_id"],
                            "base_leaf_id": result["base_leaf_id"],
                            "assigned_answer_ids": result[
                                "assigned_answer_ids"
                            ],
                            "context_unit_ids": result[
                                "context_unit_ids"
                            ],
                            "raw_text": result["raw_text"],
                            "raw_response_sha256": hashlib.sha256(
                                result["raw_text"].encode("utf-8")
                            ).hexdigest(),
                            "review_sha256": _stable_sha256(
                                result["review"]
                            ),
                        }
                        for result in context_join_results
                    ],
                }
                if shared_context_reduced is not None
                else None
            ),
            "leaf_responses": [
                {
                    "kind": result["kind"],
                    "leaf_index": result.get("leaf_index"),
                    "assigned_answer_ids": result["assigned_answer_ids"],
                    "unit_id": result.get("unit_id"),
                    "raw_text": result["raw_text"],
                    "raw_response_sha256": hashlib.sha256(
                        result["raw_text"].encode("utf-8")
                    ).hexdigest(),
                }
                for result in [*whole_results, *fragment_results]
            ],
            "answer_reducers": [
                {
                    "answer_id": result["answer_id"],
                    "status": result["status"],
                    "request_utf8_bytes": result["request_utf8_bytes"],
                    "raw_text": result["raw_text"],
                    "raw_response_sha256": hashlib.sha256(
                        result["raw_text"].encode("utf-8")
                    ).hexdigest(),
                    "review_sha256": _stable_sha256(result["review"]),
                }
                for result in answer_reducer_results
            ],
            "reducer": {
                "status": reducer_status,
                "raw_text": reducer_raw_text,
                "raw_response_sha256": hashlib.sha256(
                    reducer_raw_text.encode("utf-8")
                ).hexdigest(),
            },
            "final_review_sha256": _stable_sha256(final_review),
        },
        ensure_ascii=False,
    )
    usage = _aggregate_map_reduce_usage(
        call_records=call_records,
        partition_manifest={
            **partition_manifest,
            "answer_reducers": [
                {
                    "answer_id": result["answer_id"],
                    "status": result["status"],
                    "request_utf8_bytes": result["request_utf8_bytes"],
                    "verdict": result["review"]["verdict"],
                }
                for result in answer_reducer_results
            ],
            "reducer_request_utf8_bytes": reducer_request_bytes,
            "corpus_input_verdicts": [
                str(review.get("verdict") or "block")
                for review in source_reviews
            ],
            "fragment_leaf_verdicts": [
                str(result["review"].get("verdict") or "block")
                for result in fragment_results
            ],
            "final_verdict": final_review["verdict"],
        },
        context_envelope=context_envelope,
        final_usage=reducer_usage,
        reducer_status=reducer_status,
    )
    return final_review, raw_text, usage


async def review_analysis(
    payload: dict[str, Any],
    *,
    iteration: int,
    recovery_final: bool = False,
    audit_sink: CriticCallAuditSink | None = None,
    transport_audit_checkpoint: AuditCheckpoint | None = None,
    transport_resume_lookup: TransportResumeLookup | None = None,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Review one snapshot without imposing a corpus-length ceiling.

    Small inputs preserve the historical single-call route.  Only an input
    that cannot safely fit the current model context enters map/reduce, where
    every complete raw answer is assigned to exactly one leaf.  Leaf and
    reducer verdicts remain atomic even when their evidence was partitioned.
    """

    max_iterations, recovery_final = _critic_iteration_contract(
        iteration,
        recovery_final=recovery_final,
    )
    input_budget_bytes, context_envelope = await _critic_input_budget_bytes()
    direct_schema_name = f"aiv_analysis_critic_{iteration}"
    direct_preflight = _critic_physical_request_preflight(
        payload,
        iteration=iteration,
        max_iterations=max_iterations,
        recovery_final=recovery_final,
        schema_name=direct_schema_name,
        context_envelope=context_envelope,
    )
    reserved_output = context_envelope["reserved_output_tokens"]
    if (
        direct_preflight["request_utf8_bytes"] <= input_budget_bytes
        and direct_preflight["effective_max_completion_tokens"]
        == reserved_output
    ):
        return await _review_analysis_once(
            payload,
            iteration=iteration,
            max_iterations=max_iterations,
            recovery_final=recovery_final,
            schema_name=direct_schema_name,
            audit_sink=audit_sink,
            transport_audit_checkpoint=transport_audit_checkpoint,
            transport_resume_lookup=transport_resume_lookup,
        )
    return await _review_analysis_map_reduce(
        payload,
        iteration=iteration,
        max_iterations=max_iterations,
        recovery_final=recovery_final,
        input_budget_bytes=input_budget_bytes,
        context_envelope=context_envelope,
        audit_sink=audit_sink,
        transport_audit_checkpoint=transport_audit_checkpoint,
        transport_resume_lookup=transport_resume_lookup,
    )


def _repair_system_prompt(
    *,
    recovery_final: bool,
    fragment_leaf: bool = False,
    reducer: bool = False,
) -> str:
    prompt = CRITIC_REPAIR_REDUCE_SYSTEM if reducer else CRITIC_REPAIR_SYSTEM
    if recovery_final:
        prompt += "\n\n" + CRITIC_RECOVERY_FINAL_SUFFIX
    if fragment_leaf:
        prompt += "\n\n" + CRITIC_REPAIR_FRAGMENT_SUFFIX
    return prompt


def _repair_command(
    *,
    payload: dict[str, Any],
    incomplete_review: dict[str, Any],
    repair_context: dict[str, Any],
    iteration: int,
    max_iterations: int,
    recovery_final: bool,
    validation_errors: list[str],
    stage: str,
) -> dict[str, Any]:
    return {
        "iteration": iteration,
        "max_iterations": max_iterations,
        "recovery_final": recovery_final,
        "repair_attempt": 1,
        "max_repair_attempts": MAX_CRITIC_REPAIR_ATTEMPTS,
        "repair_stage": stage,
        "validation_errors": validation_errors,
        "incomplete_review": incomplete_review,
        "audit_payload_sha256": _stable_sha256(payload),
        "repair_context": repair_context,
    }


def _repair_messages(
    command: dict[str, Any],
    *,
    recovery_final: bool,
    fragment_leaf: bool = False,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": _repair_system_prompt(
                recovery_final=recovery_final,
                fragment_leaf=fragment_leaf,
            ),
        },
        {
            "role": "user",
            "content": json.dumps(command, ensure_ascii=False),
        },
    ]


def _repair_original_verdict_floor(
    incomplete_review: dict[str, Any],
) -> str:
    candidate = incomplete_review.get("_parsed_partial_review")
    if not isinstance(candidate, dict):
        candidate = incomplete_review
    verdict = candidate.get("verdict") if isinstance(candidate, dict) else None
    return verdict if verdict in {"pass", "revise", "block"} else "block"


def _repair_fail_closed_review(
    incomplete_review: dict[str, Any],
    *,
    reason: str,
) -> dict[str, Any]:
    """Return a schema-valid block without discarding a valid original floor."""

    block = {
        "verdict": "block",
        "summary": (
            "Автоматическое восстановление critic-решения завершилось "
            f"fail-closed: {reason}"
        ),
        "anomalies": [],
        "policy_adjustments": [],
        "annotation_guidance": "",
        "acceptance_checks": [
            "Повторить critic-проверку только после подтверждения полного "
            "lineage всех затронутых raw-ответов."
        ],
    }
    candidate = incomplete_review.get("_parsed_partial_review")
    if not isinstance(candidate, dict):
        candidate = incomplete_review
    try:
        original = _validated_review_schema(
            candidate,
            owner="Critic repair original verdict floor",
        )
    except (OpenRouterError, TypeError, ValueError):
        return _validated_review_schema(
            block,
            owner="Critic repair fail-closed fallback",
        )
    return _merge_reviews_preserving_material_findings([original], block)


def _repair_context_index(
    repair_context: dict[str, Any],
) -> dict[str, Any]:
    """Keep every affected source attested while removing raw from the index."""

    affected_index: list[dict[str, Any]] = []
    for evidence in repair_context.get("affected_answer_evidence") or []:
        if not isinstance(evidence, dict):
            continue
        raw_answer = str(evidence.get("raw_answer") or "")
        indexed = {
            key: value for key, value in evidence.items() if key != "raw_answer"
        }
        indexed.update(
            {
                "repair_source_sha256": hashlib.sha256(
                    raw_answer.encode("utf-8")
                ).hexdigest(),
                "repair_source_chars": len(raw_answer),
                "repair_source_utf8_bytes": len(raw_answer.encode("utf-8")),
            }
        )
        affected_index.append(indexed)
    return {
        **repair_context,
        "affected_answer_evidence": [],
        "affected_answer_index": affected_index,
    }


def _build_repair_fragment_plan(
    *,
    payload: dict[str, Any],
    incomplete_review: dict[str, Any],
    repair_context: dict[str, Any],
    iteration: int,
    max_iterations: int,
    recovery_final: bool,
    validation_errors: list[str],
    input_budget_bytes: int,
    context_envelope: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Partition every affected raw answer into exact, request-bounded cores."""

    indexed_context = _repair_context_index(repair_context)
    affected_evidence = [
        evidence
        for evidence in repair_context.get("affected_answer_evidence") or []
        if isinstance(evidence, dict)
    ]
    answer_ids = [
        int(evidence["answer_id"])
        for evidence in affected_evidence
        if isinstance(evidence.get("answer_id"), int)
        and not isinstance(evidence.get("answer_id"), bool)
    ]
    if len(answer_ids) != len(affected_evidence):
        raise OpenRouterError(
            "Critic repair cannot partition affected evidence without answer_id"
        )
    if len(answer_ids) != len(set(answer_ids)):
        raise OpenRouterError(
            "Critic repair affected evidence contains duplicate answer_id"
        )
    if not affected_evidence:
        raise OpenRouterError(
            "Oversized critic repair has no affected raw evidence to partition"
        )

    indexed_by_id = {
        int(evidence["answer_id"]): evidence
        for evidence in indexed_context["affected_answer_index"]
    }
    tasks: list[dict[str, Any]] = []
    answer_manifests: list[dict[str, Any]] = []
    global_index = 0
    for evidence in affected_evidence:
        answer_id = int(evidence["answer_id"])
        raw_answer = str(evidence.get("raw_answer") or "")
        target_chars = max(
            256,
            min(len(raw_answer), input_budget_bytes // 2),
        )
        while True:
            units, text_manifest = split_lossless_text(
                raw_answer,
                document_id=f"critic-repair-answer-{answer_id}",
                target_chars=target_chars,
                context_overlap_chars=DEFAULT_CONTEXT_OVERLAP_CHARS,
            )
            manifest = text_manifest.as_dict()
            if verify_units(units, manifest) != raw_answer:
                raise OpenRouterError(
                    "Critic repair fragment reconstruction changed raw evidence"
                )
            candidate_tasks: list[dict[str, Any]] = []
            request_sizes: list[int] = []
            for unit in units:
                fragment_evidence = {
                    **indexed_by_id[answer_id],
                    "raw_answer": unit.context_text,
                    "repair_raw_missing": bool(
                        evidence.get("repair_raw_missing")
                    ),
                    "repair_raw_truncated": False,
                    "repair_fragment": {
                        "version": CRITIC_MAP_REDUCE_VERSION,
                        "answer_id": answer_id,
                        "source_sha256": manifest["source_sha256"],
                        "source_chars": manifest["source_chars"],
                        "source_utf8_bytes": manifest["source_utf8_bytes"],
                        "unit_id": unit.unit_id,
                        "unit_index": unit.index,
                        "unit_count": manifest["unit_count"],
                        "core_start_char": unit.start_char,
                        "core_end_char": unit.end_char,
                        "core_sha256": unit.sha256,
                        "context_start_char": unit.context_start_char,
                        "context_end_char": unit.context_end_char,
                        "context_sha256": unit.context_sha256,
                        "core_start_in_context": unit.core_start_in_context,
                        "core_end_in_context": unit.core_end_in_context,
                        "ownership_rule": (
                            "first_decisive_evidence_character_in_core"
                        ),
                        "overlap_counts_toward_core_coverage": False,
                    },
                }
                fragment_context = {
                    **indexed_context,
                    "affected_answer_evidence": [fragment_evidence],
                    "repair_partition": {
                        "mode": "affected_answer_fragment",
                        "affected_answer_ids": answer_ids,
                        "assigned_answer_id": answer_id,
                        "assigned_unit_id": unit.unit_id,
                        "assigned_core_start_char": unit.start_char,
                        "assigned_core_end_char": unit.end_char,
                        "exact_core_accounting": True,
                    },
                }
                command = _repair_command(
                    payload=payload,
                    incomplete_review=incomplete_review,
                    repair_context=fragment_context,
                    iteration=iteration,
                    max_iterations=max_iterations,
                    recovery_final=recovery_final,
                    validation_errors=validation_errors,
                    stage="fragment_leaf",
                )
                fragment_messages = _repair_messages(
                    command,
                    recovery_final=recovery_final,
                    fragment_leaf=True,
                )
                request_bytes = _critic_messages_physical_request_utf8_bytes(
                    messages=fragment_messages,
                    schema_name=(
                        f"aiv_analysis_critic_repair_{iteration}_answer_"
                        f"{answer_id}_fragment_{unit.index}"
                    ),
                    reasoning_effort=CRITIC_REPAIR_REASONING_EFFORT,
                    temperature=0.0,
                    context_envelope=context_envelope,
                )
                candidate_tasks.append(
                    {
                        "global_index": global_index + unit.index,
                        "answer_id": answer_id,
                        "unit_id": unit.unit_id,
                        "unit_index": unit.index,
                        "command": command,
                        "request_utf8_bytes": request_bytes,
                    }
                )
                request_sizes.append(request_bytes)
            largest = max(request_sizes, default=0)
            if largest <= input_budget_bytes:
                core_ids = [unit.unit_id for unit in units]
                core_chars = sum(
                    unit.end_char - unit.start_char for unit in units
                )
                context_chars = sum(len(unit.context_text) for unit in units)
                tasks.extend(candidate_tasks)
                global_index += len(candidate_tasks)
                answer_manifests.append(
                    {
                        "answer_id": answer_id,
                        "source_sha256": manifest["source_sha256"],
                        "source_chars": manifest["source_chars"],
                        "source_utf8_bytes": manifest["source_utf8_bytes"],
                        "core_unit_ids": core_ids,
                        "core_unit_count": len(core_ids),
                        "submitted_core_chars": core_chars,
                        "submitted_context_chars": context_chars,
                        "overlap_chars_excluded_from_coverage": (
                            context_chars - core_chars
                        ),
                        "fragment_request_utf8_bytes": request_sizes,
                        "lossless_manifest": manifest,
                        "exact_core_accounting": True,
                    }
                )
                break
            if target_chars == 256:
                raise OpenRouterError(
                    "Even the minimum critic repair fragment cannot fit "
                    "without truncation: "
                    f"answer_id={answer_id}, request_bytes={largest}, "
                    f"budget_bytes={input_budget_bytes}"
                )
            ratio = max(
                0.1,
                min(0.9, input_budget_bytes / max(1, largest)),
            )
            target_chars = max(
                256,
                min(target_chars - 1, int(target_chars * ratio)),
            )

    expected_lineage = [
        {"answer_id": task["answer_id"], "unit_id": task["unit_id"]}
        for task in tasks
    ]
    return tasks, {
        "version": CRITIC_MAP_REDUCE_VERSION,
        "mode": "repair_map_reduce",
        "audit_payload_sha256": _stable_sha256(payload),
        "affected_answer_ids": answer_ids,
        "affected_answer_count": len(answer_ids),
        "fragment_count": len(tasks),
        "expected_fragment_lineage": expected_lineage,
        "answers": answer_manifests,
        "input_budget_bytes": input_budget_bytes,
        "exact_answer_accounting": True,
        "exact_fragment_accounting": True,
    }


def _aggregate_repair_usage(
    *,
    call_records: list[dict[str, Any]],
    manifest: dict[str, Any],
    status: str,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "reasoning_tokens",
        "cost",
    ):
        values = [
            record.get("usage", {}).get(key)
            for record in call_records
            if isinstance(record.get("usage"), dict)
        ]
        numeric = [
            value
            for value in values
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        if numeric:
            output[key] = sum(numeric)
    output = _critic_usage(
        output,
        recovered_from=(
            "repair_map_reduce_failure" if status == "fail_closed" else None
        ),
    )
    output["_aiv_critic_repair_map_reduce"] = {
        "version": CRITIC_MAP_REDUCE_VERSION,
        "status": status,
        "manifest": _json_safe(manifest),
        "provider_call_count": len(call_records),
        "calls": [
            {
                "kind": record.get("kind"),
                "index": record.get("index"),
                "status": record.get("status"),
                "lineage": _json_safe(record.get("lineage") or {}),
                "input_sha256": record.get("input_sha256"),
                "raw_response_sha256": (
                    hashlib.sha256(
                        str(record.get("raw_text") or "").encode("utf-8")
                    ).hexdigest()
                    if record.get("provider_response_present")
                    else None
                ),
                "raw_response_chars": len(str(record.get("raw_text") or "")),
                "usage": _json_safe(record.get("usage") or {}),
                "error_type": record.get("error_type"),
            }
            for record in call_records
        ],
    }
    return _json_safe(output)


def _repair_raw_ledger(
    *,
    manifest: dict[str, Any],
    call_records: list[dict[str, Any]],
    status: str,
    reason: str | None,
    final_review: dict[str, Any],
) -> str:
    return json.dumps(
        {
            "version": CRITIC_MAP_REDUCE_VERSION,
            "mode": "repair_map_reduce",
            "status": status,
            "reason": reason,
            "partition_manifest": manifest,
            "provider_calls": [
                {
                    "kind": record.get("kind"),
                    "index": record.get("index"),
                    "status": record.get("status"),
                    "lineage": record.get("lineage"),
                    "input_sha256": record.get("input_sha256"),
                    "raw_text": record.get("raw_text") or "",
                    "usage": record.get("usage") or {},
                    "error_type": record.get("error_type"),
                    "error_message": record.get("error_message"),
                }
                for record in call_records
            ],
            "final_review_sha256": _stable_sha256(final_review),
        },
        ensure_ascii=False,
    )


def _repair_fail_closed_result(
    incomplete_review: dict[str, Any],
    *,
    reason: str,
    manifest: dict[str, Any],
    call_records: list[dict[str, Any]],
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    review = _repair_fail_closed_review(incomplete_review, reason=reason)
    return (
        review,
        _repair_raw_ledger(
            manifest=manifest,
            call_records=call_records,
            status="fail_closed",
            reason=reason,
            final_review=review,
        ),
        _aggregate_repair_usage(
            call_records=call_records,
            manifest=manifest,
            status="fail_closed",
        ),
    )


def _repair_record_from_reduce_call(
    call: dict[str, Any],
    *,
    kind: str,
    lineage: dict[str, Any],
) -> dict[str, Any]:
    return {
        "kind": kind,
        "index": int(call["level"]) * 100_000 + int(call["group_index"]),
        "status": str(call.get("status") or "completed"),
        "lineage": lineage,
        "input_sha256": _stable_sha256(call.get("input") or {}),
        "raw_text": str(call.get("raw_text") or ""),
        "usage": dict(call.get("usage") or {}),
        "provider_response_present": True,
        "error_type": None,
        "error_message": None,
    }


async def _repair_analysis_review_map_reduce(
    payload: dict[str, Any],
    incomplete_review: dict[str, Any],
    *,
    repair_context: dict[str, Any],
    iteration: int,
    max_iterations: int,
    recovery_final: bool,
    validation_errors: list[str],
    input_budget_bytes: int,
    context_envelope: dict[str, Any],
    audit_sink: CriticCallAuditSink | None,
    transport_audit_checkpoint: AuditCheckpoint | None = None,
    transport_resume_lookup: TransportResumeLookup | None = None,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    try:
        fragment_tasks, manifest = _build_repair_fragment_plan(
            payload=payload,
            incomplete_review=incomplete_review,
            repair_context=repair_context,
            iteration=iteration,
            max_iterations=max_iterations,
            recovery_final=recovery_final,
            validation_errors=validation_errors,
            input_budget_bytes=input_budget_bytes,
            context_envelope=context_envelope,
        )
    except (OpenRouterError, TypeError, ValueError) as exc:
        return _repair_fail_closed_result(
            incomplete_review,
            reason=str(exc),
            manifest={
                "version": CRITIC_MAP_REDUCE_VERSION,
                "mode": "repair_map_reduce_planning_failure",
                "audit_payload_sha256": _stable_sha256(payload),
                "input_budget_bytes": input_budget_bytes,
                "context_envelope": _json_safe(context_envelope),
            },
            call_records=[],
        )

    semaphore = asyncio.Semaphore(CRITIC_MAP_CONCURRENCY)

    async def run_fragment(task: dict[str, Any]) -> dict[str, Any]:
        response: Any = None
        command = task["command"]
        lineage = {
            "answer_id": int(task["answer_id"]),
            "unit_id": str(task["unit_id"]),
            "unit_index": int(task["unit_index"]),
        }
        schema_name = (
            f"aiv_analysis_critic_repair_{iteration}_answer_"
            f"{task['answer_id']}_fragment_{task['unit_index']}"
        )
        messages = _repair_messages(
            command,
            recovery_final=recovery_final,
            fragment_leaf=True,
        )
        descriptor = _critic_logical_call_descriptor(
            iteration=iteration,
            kind="repair_fragment_leaf",
            index=int(task["global_index"]),
            schema_name=schema_name,
            call_input=command,
            lineage=lineage,
        )
        attempt_id = _critic_attempt_id(descriptor)
        cached = await _lookup_completed_critic_call(audit_sink, descriptor)
        if cached is not None:
            review, raw_text, usage = cached
            return {
                "kind": "repair_fragment_leaf",
                "index": int(task["global_index"]),
                "status": "reused",
                "lineage": lineage,
                "input_sha256": _stable_sha256(command),
                "raw_text": raw_text,
                "usage": usage,
                "provider_response_present": True,
                "error_type": None,
                "error_message": None,
                "review": review,
            }
        try:
            async with semaphore:
                response = await _critic_atomic_chat(
                    messages=messages,
                    schema_name=schema_name,
                    reasoning_effort=CRITIC_REPAIR_REASONING_EFFORT,
                    temperature=0.0,
                    transport_audit_checkpoint=transport_audit_checkpoint,
                    transport_resume_lookup=transport_resume_lookup,
                )
            if not isinstance(response.parsed, dict):
                raise OpenRouterError(
                    "Analysis critic repair fragment returned no verdict"
                )
            review = _validated_review_schema(
                response.parsed,
                owner=(
                    "Critic repair fragment "
                    f"{task['answer_id']}:{task['unit_index']}"
                ),
            )
        except BaseException as exc:
            result = getattr(exc, "result", None) or response
            raw_text = str(getattr(result, "text", "") or "")
            usage = (
                _critic_usage(dict(getattr(result, "usage", {}) or {}))
                if result is not None
                else {}
            )
            await _emit_critic_call_audit(
                audit_sink,
                attempt_id=attempt_id,
                iteration=iteration,
                kind="repair_fragment_leaf",
                index=int(task["global_index"]),
                call_input=command,
                lineage=lineage,
                status=(
                    "cancelled"
                    if isinstance(exc, asyncio.CancelledError)
                    else "failed"
                ),
                review=None,
                raw_text=raw_text,
                usage=usage,
                provider_response_present=result is not None,
                error=exc,
                schema_name=schema_name,
            )
            if isinstance(exc, asyncio.CancelledError):
                raise
            return {
                "kind": "repair_fragment_leaf",
                "index": int(task["global_index"]),
                "status": "failed",
                "lineage": lineage,
                "input_sha256": _stable_sha256(command),
                "raw_text": raw_text,
                "usage": usage,
                "provider_response_present": result is not None,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "review": None,
            }

        raw_text = response.text
        usage = _critic_usage(response.usage)
        await _emit_critic_call_audit(
            audit_sink,
            attempt_id=attempt_id,
            iteration=iteration,
            kind="repair_fragment_leaf",
            index=int(task["global_index"]),
            call_input=command,
            lineage=lineage,
            status="completed",
            review=review,
            raw_text=raw_text,
            usage=usage,
            provider_response_present=True,
            error=None,
            schema_name=schema_name,
        )
        return {
            "kind": "repair_fragment_leaf",
            "index": int(task["global_index"]),
            "status": "completed",
            "lineage": lineage,
            "input_sha256": _stable_sha256(command),
            "raw_text": raw_text,
            "usage": usage,
            "provider_response_present": True,
            "error_type": None,
            "error_message": None,
            "review": review,
        }

    map_tasks = [
        asyncio.create_task(
            run_fragment(task),
            name=(
                f"aiv-critic-repair-fragment-{iteration}-"
                f"{task['answer_id']}-{task['unit_index']}"
            ),
        )
        for task in fragment_tasks
    ]
    try:
        map_outcomes = await asyncio.gather(
            *map_tasks,
            return_exceptions=True,
        )
    except asyncio.CancelledError:
        for task in map_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*map_tasks, return_exceptions=True)
        raise

    map_records = [
        outcome for outcome in map_outcomes if isinstance(outcome, dict)
    ]
    map_task_failures = [
        outcome
        for outcome in map_outcomes
        if isinstance(outcome, BaseException)
    ]
    failed_records = [
        record for record in map_records if record.get("status") != "completed"
    ]
    if map_task_failures or failed_records:
        return _repair_fail_closed_result(
            incomplete_review,
            reason=(
                "Не все fragment-leaf вызовы ремонта завершились: "
                f"task_failures={len(map_task_failures)}, "
                f"failed_calls={len(failed_records)}"
            ),
            manifest=manifest,
            call_records=map_records,
        )

    expected_fragment_lineage = manifest["expected_fragment_lineage"]
    actual_fragment_lineage = [
        {
            "answer_id": int(record["lineage"]["answer_id"]),
            "unit_id": str(record["lineage"]["unit_id"]),
        }
        for record in map_records
    ]
    if actual_fragment_lineage != expected_fragment_lineage or len(
        {_review_reduce_lineage_token(value) for value in actual_fragment_lineage}
    ) != len(actual_fragment_lineage):
        return _repair_fail_closed_result(
            incomplete_review,
            reason="Fragment lineage ремонта неполон или содержит дубли",
            manifest=manifest,
            call_records=map_records,
        )

    records_by_answer: dict[int, list[dict[str, Any]]] = {}
    for record in map_records:
        answer_id = int(record["lineage"]["answer_id"])
        records_by_answer.setdefault(answer_id, []).append(record)
    answer_manifest_by_id = {
        int(value["answer_id"]): value for value in manifest["answers"]
    }
    indexed_context = _repair_context_index(repair_context)
    affected_index_by_id = {
        int(value["answer_id"]): value
        for value in indexed_context["affected_answer_index"]
    }

    async def reduce_answer(answer_id: int) -> dict[str, Any]:
        fragment_records = records_by_answer[answer_id]
        answer_manifest = answer_manifest_by_id[answer_id]
        expected_units = [
            str(value) for value in answer_manifest["core_unit_ids"]
        ]

        def build_payload(
            nodes: list[dict[str, Any]],
            level: int,
            group_index: int,
        ) -> dict[str, Any]:
            lineage = [
                str(value) for node in nodes for value in node["lineage"]
            ]
            return {
                "version": CRITIC_MAP_REDUCE_VERSION,
                "stage": "repair_answer_reduce",
                "iteration": iteration,
                "recovery_final": recovery_final,
                "audit_payload_sha256": _stable_sha256(payload),
                "validation_errors": validation_errors,
                "incomplete_review_sha256": _stable_sha256(incomplete_review),
                "original_verdict_floor": _repair_original_verdict_floor(
                    incomplete_review
                ),
                "answer_id": answer_id,
                "affected_answer_index": affected_index_by_id[answer_id],
                "fragment_manifest": {
                    "manifest_sha256": _stable_sha256(answer_manifest),
                    "source_sha256": answer_manifest["source_sha256"],
                    "source_chars": answer_manifest["source_chars"],
                    "core_unit_count": len(expected_units),
                    "group_lineage": _review_reduce_lineage_descriptor(lineage),
                    "exact_core_accounting": True,
                },
                "reduction_tree": {
                    "level": level,
                    "group_index": group_index,
                    "lineage": (
                        lineage
                        if level == 0
                        else _review_reduce_lineage_descriptor(lineage)
                    ),
                },
                "fragment_reviews": [
                    {
                        "unit_lineage": _review_reduce_lineage_descriptor(
                            list(node["lineage"])
                        ),
                        "review_sha256": _stable_sha256(node["review"]),
                        "review": node["review"],
                    }
                    for node in nodes
                ],
                "verdict_floor": _merge_reviews_preserving_material_findings(
                    [node["review"] for node in nodes],
                    _critic_review_pass_seed(
                        "Кодовый verdict floor repair-фрагментов."
                    ),
                )["verdict"],
            }

        reduced = await _hierarchical_review_reduce(
            source_nodes=[
                {
                    "lineage": [str(record["lineage"]["unit_id"])],
                    "review": record["review"],
                }
                for record in fragment_records
            ],
            expected_lineage=expected_units,
            build_payload=build_payload,
            system_prompt=_repair_system_prompt(
                recovery_final=recovery_final,
                reducer=True,
            ),
            schema_name_prefix=(
                f"aiv_analysis_critic_repair_{iteration}_answer_"
                f"{answer_id}_reducer"
            ),
            owner=f"Critic repair answer {answer_id} reducer",
            audit_kind="repair_answer_reducer",
            iteration=iteration,
            recovery_final=recovery_final,
            input_budget_bytes=input_budget_bytes,
            context_envelope=context_envelope,
            audit_sink=audit_sink,
            transport_audit_checkpoint=transport_audit_checkpoint,
            transport_resume_lookup=transport_resume_lookup,
            collapse_answer_duplicates=True,
        )
        return {
            "answer_id": answer_id,
            "lineage": [answer_id],
            "review": reduced["review"],
            "provider_calls": list(reduced["provider_calls"]),
            "tree_levels": reduced["tree_levels"],
            "status": reduced["status"],
        }

    answer_ids = [int(value) for value in manifest["affected_answer_ids"]]
    answer_tasks = [
        asyncio.create_task(
            reduce_answer(answer_id),
            name=f"aiv-critic-repair-answer-reducer-{iteration}-{answer_id}",
        )
        for answer_id in answer_ids
    ]
    try:
        answer_outcomes = await asyncio.gather(
            *answer_tasks,
            return_exceptions=True,
        )
    except asyncio.CancelledError:
        for task in answer_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*answer_tasks, return_exceptions=True)
        raise

    answer_results = [
        outcome for outcome in answer_outcomes if isinstance(outcome, dict)
    ]
    answer_failures = [
        outcome
        for outcome in answer_outcomes
        if isinstance(outcome, BaseException)
    ]
    call_records = list(map_records)
    for result in answer_results:
        for call in result["provider_calls"]:
            call_records.append(
                _repair_record_from_reduce_call(
                    call,
                    kind="repair_answer_reducer",
                    lineage={
                        "answer_id": int(result["answer_id"]),
                        "unit_ids": list(call["lineage"]),
                        "tree_level": int(call["level"]),
                    },
                )
            )
    if answer_failures or [
        int(result["answer_id"]) for result in answer_results
    ] != answer_ids:
        return _repair_fail_closed_result(
            incomplete_review,
            reason=(
                "Не все answer-reducers ремонта завершились: "
                f"failures={len(answer_failures)}"
            ),
            manifest=manifest,
            call_records=call_records,
        )

    def build_corpus_payload(
        nodes: list[dict[str, Any]],
        level: int,
        group_index: int,
    ) -> dict[str, Any]:
        lineage = [
            int(value) for node in nodes for value in node["lineage"]
        ]
        return {
            "version": CRITIC_MAP_REDUCE_VERSION,
            "stage": "repair_corpus_reduce",
            "iteration": iteration,
            "recovery_final": recovery_final,
            "audit_payload_sha256": _stable_sha256(payload),
            "validation_errors": validation_errors,
            "incomplete_review_sha256": _stable_sha256(incomplete_review),
            "original_verdict_floor": _repair_original_verdict_floor(
                incomplete_review
            ),
            "partition_manifest_sha256": _stable_sha256(manifest),
            "affected_answer_count": len(answer_ids),
            "reduction_tree": {
                "level": level,
                "group_index": group_index,
                "lineage": (
                    lineage
                    if level == 0
                    else _review_reduce_lineage_descriptor(lineage)
                ),
            },
            "answer_reviews": [
                {
                    "answer_lineage": _review_reduce_lineage_descriptor(
                        list(node["lineage"])
                    ),
                    "review_sha256": _stable_sha256(node["review"]),
                    "review": node["review"],
                }
                for node in nodes
            ],
            "verdict_floor": _merge_reviews_preserving_material_findings(
                [node["review"] for node in nodes],
                _critic_review_pass_seed(
                    "Кодовый verdict floor repair-ответов."
                ),
            )["verdict"],
        }

    try:
        corpus_reduced = await _hierarchical_review_reduce(
            source_nodes=[
                {
                    "lineage": [int(result["answer_id"])],
                    "review": result["review"],
                }
                for result in answer_results
            ],
            expected_lineage=answer_ids,
            build_payload=build_corpus_payload,
            system_prompt=_repair_system_prompt(
                recovery_final=recovery_final,
                reducer=True,
            ),
            schema_name_prefix=(
                f"aiv_analysis_critic_repair_{iteration}_corpus_reducer"
            ),
            owner="Critic repair corpus reducer",
            audit_kind="repair_corpus_reducer",
            iteration=iteration,
            recovery_final=recovery_final,
            input_budget_bytes=input_budget_bytes,
            context_envelope=context_envelope,
            audit_sink=audit_sink,
            transport_audit_checkpoint=transport_audit_checkpoint,
            transport_resume_lookup=transport_resume_lookup,
        )
    except asyncio.CancelledError:
        raise
    except BaseException as exc:
        result = getattr(exc, "result", None)
        if result is not None:
            call_records.append(
                {
                    "kind": "repair_corpus_reducer",
                    "index": -1,
                    "status": "failed",
                    "lineage": {"answer_ids": answer_ids},
                    "input_sha256": None,
                    "raw_text": str(getattr(result, "text", "") or ""),
                    "usage": _critic_usage(
                        dict(getattr(result, "usage", {}) or {})
                    ),
                    "provider_response_present": True,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
        return _repair_fail_closed_result(
            incomplete_review,
            reason=f"Corpus reducer ремонта завершился ошибкой: {exc}",
            manifest=manifest,
            call_records=call_records,
        )

    for call in corpus_reduced["provider_calls"]:
        call_records.append(
            _repair_record_from_reduce_call(
                call,
                kind="repair_corpus_reducer",
                lineage={
                    "answer_ids": list(call["lineage"]),
                    "tree_level": int(call["level"]),
                },
            )
        )
    repaired = corpus_reduced["review"]
    omitted_ids = repair_context["evidence_limits"][
        "omitted_referenced_answer_ids"
    ]
    missing_ids = [
        evidence.get("answer_id")
        for evidence in repair_context["affected_answer_evidence"]
        if evidence.get("repair_raw_missing") is True
    ]
    if (omitted_ids or missing_ids) and repaired.get("verdict") != "block":
        return _repair_fail_closed_result(
            incomplete_review,
            reason=(
                "Analysis critic repair used incomplete affected evidence; "
                "only block is safe"
            ),
            manifest=manifest,
            call_records=call_records,
        )
    return (
        repaired,
        _repair_raw_ledger(
            manifest=manifest,
            call_records=call_records,
            status="completed",
            reason=None,
            final_review=repaired,
        ),
        _aggregate_repair_usage(
            call_records=call_records,
            manifest={
                **manifest,
                "context_envelope": _json_safe(context_envelope),
                "answer_reducer_statuses": [
                    {
                        "answer_id": result["answer_id"],
                        "status": result["status"],
                        "tree_levels": result["tree_levels"],
                    }
                    for result in answer_results
                ],
                "corpus_reducer_status": corpus_reduced["status"],
                "final_verdict": repaired["verdict"],
            },
            status="completed",
        ),
    )


async def repair_analysis_review(
    payload: dict[str, Any],
    incomplete_review: dict[str, Any],
    *,
    iteration: int,
    validation_errors: list[str],
    recovery_final: bool = False,
    audit_sink: CriticCallAuditSink | None = None,
    transport_audit_checkpoint: AuditCheckpoint | None = None,
    transport_resume_lookup: TransportResumeLookup | None = None,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Repair one decision without imposing a length ceiling on raw evidence.

    A request that fits the live model envelope preserves the historical
    single-call route. Larger affected raw answers are split losslessly and
    reduced through bounded fan-in trees with exact code-owned lineage. Every
    paid call is handed to ``audit_sink`` before any sibling/reducer can fail.
    Each repair verdict is atomic; output-limited JSON fails closed instead of
    being promoted through a continuation.
    """

    max_iterations, recovery_final = _critic_iteration_contract(
        iteration,
        recovery_final=recovery_final,
    )
    repair_context = _compact_repair_context(payload, incomplete_review)
    direct_command = _repair_command(
        payload=payload,
        incomplete_review=incomplete_review,
        repair_context=repair_context,
        iteration=iteration,
        max_iterations=max_iterations,
        recovery_final=recovery_final,
        validation_errors=validation_errors,
        stage="direct",
    )
    direct_messages = _repair_messages(
        direct_command,
        recovery_final=recovery_final,
    )
    direct_request_bytes = _json_utf8_bytes(direct_messages)
    context_envelope: dict[str, Any] = {
        "input_budget_bytes": CRITIC_SMALL_SINGLE_CALL_BYTES,
        "source": "small_single_call_threshold",
    }
    input_budget_bytes = CRITIC_SMALL_SINGLE_CALL_BYTES
    if direct_request_bytes > CRITIC_SMALL_SINGLE_CALL_BYTES:
        try:
            input_budget_bytes, context_envelope = (
                await _critic_input_budget_bytes()
            )
            direct_request_bytes = (
                _critic_messages_physical_request_utf8_bytes(
                    messages=direct_messages,
                    schema_name=f"aiv_analysis_critic_repair_{iteration}",
                    reasoning_effort=CRITIC_REPAIR_REASONING_EFFORT,
                    temperature=0.0,
                    context_envelope=context_envelope,
                )
            )
        except (OpenRouterError, TypeError, ValueError) as exc:
            return _repair_fail_closed_result(
                incomplete_review,
                reason=f"Не удалось определить безопасное окно repair: {exc}",
                manifest={
                    "version": CRITIC_MAP_REDUCE_VERSION,
                    "mode": "repair_context_envelope_failure",
                    "audit_payload_sha256": _stable_sha256(payload),
                    "direct_request_utf8_bytes": direct_request_bytes,
                },
                call_records=[],
            )
    if direct_request_bytes > input_budget_bytes:
        return await _repair_analysis_review_map_reduce(
            payload,
            incomplete_review,
            repair_context=repair_context,
            iteration=iteration,
            max_iterations=max_iterations,
            recovery_final=recovery_final,
            validation_errors=validation_errors,
            input_budget_bytes=input_budget_bytes,
            context_envelope=context_envelope,
            audit_sink=audit_sink,
            transport_audit_checkpoint=transport_audit_checkpoint,
            transport_resume_lookup=transport_resume_lookup,
        )

    schema_name = f"aiv_analysis_critic_repair_{iteration}"
    lineage = {
        "affected_answer_ids": [
            evidence.get("answer_id")
            for evidence in repair_context["affected_answer_evidence"]
        ]
    }
    descriptor = _critic_logical_call_descriptor(
        iteration=iteration,
        kind="compact_repair",
        index=0,
        schema_name=schema_name,
        call_input=direct_command,
        lineage=lineage,
    )
    attempt_id = _critic_attempt_id(descriptor)
    cached = await _lookup_completed_critic_call(audit_sink, descriptor)
    response: Any = None
    if cached is not None:
        repaired, final_raw_text, usage = cached
    else:
        try:
            response = await _critic_atomic_chat(
                messages=direct_messages,
                schema_name=schema_name,
                reasoning_effort=CRITIC_REPAIR_REASONING_EFFORT,
                temperature=0.0,
                transport_audit_checkpoint=transport_audit_checkpoint,
                transport_resume_lookup=transport_resume_lookup,
            )
            if not isinstance(response.parsed, dict):
                raise OpenRouterError(
                    "Analysis critic repair returned no structured verdict"
                )
            repaired = _normalize_review(response.parsed)
        except BaseException as exc:
            result = getattr(exc, "result", None) or response
            raw_text = str(getattr(result, "text", "") or "")
            usage = (
                _critic_usage(dict(getattr(result, "usage", {}) or {}))
                if result is not None
                else {}
            )
            await _emit_critic_call_audit(
                audit_sink,
                attempt_id=attempt_id,
                iteration=iteration,
                kind="compact_repair",
                index=0,
                call_input=direct_command,
                lineage=lineage,
                status=(
                    "cancelled"
                    if isinstance(exc, asyncio.CancelledError)
                    else "failed"
                ),
                review=None,
                raw_text=raw_text,
                usage=usage,
                provider_response_present=result is not None,
                error=exc,
                schema_name=schema_name,
            )
            if isinstance(exc, asyncio.CancelledError):
                raise
            record = {
                "kind": "compact_repair",
                "index": 0,
                "status": "failed",
                "lineage": lineage,
                "input_sha256": _stable_sha256(direct_command),
                "raw_text": raw_text,
                "usage": usage,
                "provider_response_present": result is not None,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            return _repair_fail_closed_result(
                incomplete_review,
                reason=f"Compact repair завершился ошибкой: {exc}",
                manifest={
                    "version": CRITIC_MAP_REDUCE_VERSION,
                    "mode": "direct_repair",
                    "audit_payload_sha256": _stable_sha256(payload),
                    "direct_request_utf8_bytes": direct_request_bytes,
                    "input_budget_bytes": input_budget_bytes,
                },
                call_records=[record],
            )

        usage = _critic_usage(response.usage)
        final_raw_text = response.text
        await _emit_critic_call_audit(
            audit_sink,
            attempt_id=attempt_id,
            iteration=iteration,
            kind="compact_repair",
            index=0,
            call_input=direct_command,
            lineage=lineage,
            status="completed",
            review=repaired,
            raw_text=final_raw_text,
            usage=usage,
            provider_response_present=True,
            error=None,
            schema_name=schema_name,
        )
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
    return repaired, final_raw_text, usage
