"""Rebuild a completed report without rerunning or re-annotating the model panel.

This operator tool is intentionally narrower than ``reprocess_saved_run.py``:
it reuses the saved site profile, entity catalog, raw answers and answer
annotations.  Only the deterministic metrics, technical review, final
editorial layer and illustrations are rebuilt.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy import func, select

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from app.db import SessionLocal
from app.models import (
    AnswerAnnotation,
    ModelAnswer,
    ProbeType,
    Run,
    RunArtifact,
    RunStatus,
    SitePage,
)
from app.services import analyzer, crawler
from app.services.publication_contract import (
    PublicationContractError,
    ensure_publication_contract,
    publication_snapshot,
    publication_snapshot_digest,
    replace_completed_publication,
)
from app.services.site_preview import get_saved_site_preview


class RebuildGuardError(RuntimeError):
    """The saved run does not satisfy the no-panel rebuild contract."""


def canonical_run_id(value: str) -> str:
    try:
        return str(uuid.UUID(value))
    except (AttributeError, TypeError, ValueError) as exc:
        raise RebuildGuardError("run_id должен быть корректным UUID.") from exc


async def _artifact_dict(run_id: str, artifact_key: str) -> dict[str, Any]:
    async with SessionLocal() as session:
        artifact = (
            await session.execute(
                select(RunArtifact).where(
                    RunArtifact.run_id == run_id,
                    RunArtifact.artifact_key == artifact_key,
                    RunArtifact.status == "completed",
                )
            )
        ).scalar_one_or_none()
    if artifact is None or not isinstance(artifact.output_json, dict):
        raise RebuildGuardError(
            f"Не найден завершённый артефакт {artifact_key}."
        )
    return dict(artifact.output_json)


async def _optional_artifact_dict(
    run_id: str,
    artifact_key: str,
) -> dict[str, Any] | None:
    async with SessionLocal() as session:
        artifact = (
            await session.execute(
                select(RunArtifact).where(
                    RunArtifact.run_id == run_id,
                    RunArtifact.artifact_key == artifact_key,
                    RunArtifact.status == "completed",
                )
            )
        ).scalar_one_or_none()
    if artifact is None or not isinstance(artifact.output_json, dict):
        return None
    return dict(artifact.output_json)


async def _validate_saved_inputs(run_id: str) -> str:
    async with SessionLocal() as session:
        run = (
            await session.execute(select(Run).where(Run.id == run_id))
        ).scalar_one_or_none()
        if run is None:
            raise RebuildGuardError(f"Проверка {run_id} не найдена.")
        if run.status != RunStatus.completed:
            raise RebuildGuardError(
                "Пересборка разрешена только для завершённой проверки."
            )
        try:
            receipt = await ensure_publication_contract(
                session,
                run,
                allow_legacy_baseline=False,
            )
        except PublicationContractError as exc:
            raise RebuildGuardError(
                "Текущий отчёт не прошёл проверку контракта публикации; "
                "используйте полный reprocess_saved_run.py."
            ) from exc
        if not isinstance(receipt, dict) or receipt.get("legacy_baseline") is True:
            raise RebuildGuardError(
                "Legacy-отчёт нельзя точечно пересобирать без подтверждённого "
                "reader-copy provenance; используйте полный reprocess_saved_run.py."
            )
        expected_snapshot_digest = publication_snapshot_digest(
            publication_snapshot(
                report_json=run.report_json,
                analysis_markdown=run.analysis_markdown,
            )
        )
        answer_count = int(
            (
                await session.execute(
                    select(func.count(ModelAnswer.id)).where(
                        ModelAnswer.run_id == run_id,
                        ModelAnswer.status == "completed",
                        ModelAnswer.response_text.is_not(None),
                        func.length(func.trim(ModelAnswer.response_text)) > 0,
                    )
                )
            ).scalar_one()
        )
        annotation_count = int(
            (
                await session.execute(
                    select(func.count(AnswerAnnotation.id))
                    .join(
                        ModelAnswer,
                        AnswerAnnotation.answer_id == ModelAnswer.id,
                    )
                    .where(
                        ModelAnswer.run_id == run_id,
                        ModelAnswer.status == "completed",
                    )
                )
            ).scalar_one()
        )
    if answer_count < 1 or annotation_count != answer_count:
        raise RebuildGuardError(
            "Нужен полный комплект сохранённых ответов и аннотаций: "
            f"{answer_count} ответов, {annotation_count} аннотаций."
        )
    return expected_snapshot_digest


async def _refresh_control_pages(run_id: str) -> list[dict[str, Any]]:
    async with SessionLocal() as session:
        run = (
            await session.execute(select(Run).where(Run.id == run_id))
        ).scalar_one()
        pages = list(
            (
                await session.execute(
                    select(SitePage)
                    .where(SitePage.run_id == run_id)
                    .order_by(SitePage.id)
                )
            )
            .scalars()
            .all()
        )

    refreshed: list[dict[str, Any]] = []
    for index, page in enumerate(pages):
        probe = await crawler._do_probe(
            url=page.url,
            user_agent=crawler.USER_AGENT_STRINGS["Chrome-control"],
            timeout_seconds=max(20, settings.DEFAULT_TIMEOUT_SECONDS),
            concurrency=1,
            proxy_url=None,
        )
        if (
            probe.error_class
            or probe.http_status is None
            or not 200 <= probe.http_status < 400
            or probe.body_truncated
        ):
            raise RebuildGuardError(
                "Полное контрольное чтение не удалось: "
                f"{page.url} · status={probe.http_status} · "
                f"error={probe.error_class} · truncated={probe.body_truncated}"
            )
        job = crawler._Job(
            domain=str(run.domain or ""),
            user_agent_label="Chrome-control",
            user_agent_string=crawler.USER_AGENT_STRINGS["Chrome-control"],
            target_url=page.url,
            probe_type=ProbeType.main_page,
            page_kind=page.page_kind,
        )
        await crawler._persist_probe(
            run_id,
            job,
            probe,
            {
                "proxy_used": False,
                "country": None,
                "fallback_direct": False,
                "operator_refresh": True,
            },
        )
        await crawler._store_site_page(
            run_id,
            url=page.url,
            page_kind=page.page_kind or "other",
            probe=probe,
        )
        refreshed.append(
            {
                "url": page.url,
                "bytes": probe.response_size_bytes,
                "status": probe.http_status,
            }
        )
        if index + 1 < len(pages):
            await asyncio.sleep(crawler.MIN_GAP_PER_DOMAIN_SECONDS)
    return refreshed


async def rebuild_from_saved_annotations(
    value: str,
    *,
    refresh_control_pages: bool = False,
) -> dict[str, Any]:
    run_id = canonical_run_id(value)
    expected_snapshot_digest = await _validate_saved_inputs(run_id)

    refreshed = (
        await _refresh_control_pages(run_id)
        if refresh_control_pages
        else []
    )
    profile = await _artifact_dict(run_id, "site_profile")
    catalog = await _artifact_dict(run_id, "entity_catalog")
    critic_gate = await _artifact_dict(run_id, "analysis_critic_gate")
    if critic_gate.get("passed") is not True:
        raise RebuildGuardError(
            "Сохранённая аналитика не прошла независимого критика."
        )
    policy_history = [
        dict(step)
        for step in critic_gate.get("policy_history") or []
        if isinstance(step, dict)
    ]
    critic_policy = await _optional_artifact_dict(
        run_id,
        "analysis_critic_policy",
    )
    annotation_context_payload = critic_gate.get("annotation_context")
    if policy_history or annotation_context_payload is not None:
        policy_steps = (
            critic_policy.get("policy_history")
            if isinstance(critic_policy, dict)
            else None
        )
        exact_policy = policy_steps == policy_history
        recovery_suffix = (
            isinstance(policy_steps, list)
            and len(policy_history) == len(policy_steps) + 1
            and policy_history[: len(policy_steps)] == policy_steps
            and isinstance(policy_history[-1], dict)
            and policy_history[-1].get("kind")
            == "orchestrated_targeted_annotation_repair"
            and isinstance(critic_policy.get("executing_recovery"), dict)
            and critic_policy["executing_recovery"].get("orchestrator_epoch")
            == policy_history[-1].get("orchestrator_epoch")
            and critic_policy["executing_recovery"].get("target_answer_ids")
            == policy_history[-1].get("target_answer_ids")
        )
        if (
            not isinstance(critic_policy, dict)
            or not isinstance(critic_policy.get("effective_catalog"), dict)
            or (
                annotation_context_payload is not None
                and not isinstance(
                    critic_policy.get("effective_profile"),
                    dict,
                )
            )
            or not (exact_policy or recovery_suffix)
            or (
                annotation_context_payload is not None
                and (
                    critic_policy.get("annotation_context")
                    != annotation_context_payload
                    or critic_policy.get("annotation_context_sha256")
                    != critic_gate.get("annotation_context_sha256")
                )
            )
        ):
            raise RebuildGuardError(
                "Не найдена политика, подтверждённая текущим critic gate."
            )
        if annotation_context_payload is not None:
            profile = dict(critic_policy["effective_profile"])
        catalog = dict(critic_policy["effective_catalog"])
    research_guidance = analyzer._critic_policy_guidance(policy_history)
    raw_rows = await analyzer._metric_rows(
        run_id,
        annotation_input_sha256="saved-rebuild-raw-receipt-validation",
    )
    if annotation_context_payload is not None:
        try:
            annotation_context = analyzer._validated_annotation_context_manifest(
                annotation_context_payload,
                rows=raw_rows,
                profile=profile,
                catalog=catalog,
            )
        except analyzer.OrchestratorContractError as exc:
            raise RebuildGuardError(
                "Annotation context critic gate не совпадает с raw."
            ) from exc
        if (
            critic_gate.get("annotation_context_sha256")
            != analyzer._stable_json_sha256(annotation_context)
        ):
            raise RebuildGuardError(
                "Digest annotation context critic gate не совпадает."
            )
        annotation_input_sha256 = annotation_context[
            "annotation_input_sha256"
        ]
    else:
        # Historical gates predate the explicit annotation-context receipt.
        # Keep their previous deterministic reconstruction path intact.
        receipt_payload = critic_gate.get("target_mention_receipts", [])
        if not isinstance(receipt_payload, list):
            raise RebuildGuardError("Некорректный receipt manifest critic gate.")
        target_mention_receipts = (
            analyzer._validated_target_mention_receipt_manifest(
                receipt_payload,
                rows=raw_rows,
                profile=profile,
                catalog=catalog,
            )
        )
        if "target_mention_receipts_sha256" in critic_gate and (
            analyzer._stable_json_sha256(target_mention_receipts)
            != critic_gate.get("target_mention_receipts_sha256")
        ):
            raise RebuildGuardError(
                "Receipt manifest critic gate не совпадает с raw."
            )
        annotation_input_sha256 = analyzer._annotation_context_sha256(
            profile,
            catalog,
            research_guidance,
            target_mention_receipts=target_mention_receipts,
        )
    stored_digest_map = critic_gate.get("annotation_input_sha256_by_answer_id")
    annotation_input_sha256_by_answer_id = None
    if stored_digest_map is not None:
        if not isinstance(stored_digest_map, dict):
            raise RebuildGuardError("Некорректная карта annotation digest critic gate.")
        try:
            annotation_input_sha256_by_answer_id = {
                int(answer_id): str(digest)
                for answer_id, digest in stored_digest_map.items()
            }
        except (TypeError, ValueError) as exc:
            raise RebuildGuardError(
                "Некорректная карта annotation digest critic gate."
            ) from exc
        if (
            any(answer_id <= 0 for answer_id in annotation_input_sha256_by_answer_id)
            or any(
                len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
                for digest in annotation_input_sha256_by_answer_id.values()
            )
            or {
                str(answer_id): digest
                for answer_id, digest in sorted(
                    annotation_input_sha256_by_answer_id.items()
                )
            }
            != stored_digest_map
        ):
            raise RebuildGuardError(
                "Некорректная карта annotation digest critic gate."
            )
    rows = await analyzer._metric_rows(
        run_id,
        annotation_input_sha256=annotation_input_sha256,
        annotation_input_sha256_by_answer_id=(
            annotation_input_sha256_by_answer_id
        ),
    )
    if (
        annotation_input_sha256_by_answer_id is not None
        and analyzer._current_annotation_input_digests(rows)
        != annotation_input_sha256_by_answer_id
    ):
        raise RebuildGuardError(
            "Сохранённые аннотации не совпадают с картой critic gate."
        )
    metrics = analyzer._compute_metrics(rows, profile, catalog)
    expected_corpus_cells = await analyzer._expected_corpus_cells(
        run_id,
        rows,
    )
    corpus_manifest = analyzer._final_corpus_manifest(
        rows,
        expected_cells=expected_corpus_cells,
    )
    provenance = analyzer._critic_provenance_digests(
        profile=profile,
        catalog=catalog,
        rows=rows,
        metrics=metrics,
        policy_history=policy_history,
    )
    if provenance != critic_gate.get("provenance"):
        raise RebuildGuardError(
            "Профиль, каталог, сценарии, raw-ответы, аннотации или метрики "
            "изменились после critic gate; "
            "нужен переанализ сохранённых raw-ответов."
        )
    if (
        not corpus_manifest["complete"]
        or corpus_manifest.get("digest")
        != (critic_gate.get("corpus_manifest") or {}).get("digest")
    ):
        raise RebuildGuardError(
            "Полный корпус ответов не совпадает с manifest critic gate; "
            "нужен переанализ сохранённых raw-ответов."
        )
    technical = await analyzer._technical_summary(run_id)
    technical_review = await analyzer._review_technical_summary(
        run_id,
        technical,
    )
    await analyzer._save_artifact(
        run_id,
        stage_key="knowledge_gap",
        artifact_key="metrics",
        status="completed",
        model=None,
        prompt_version=analyzer.METRICS_VERSION,
        input_json={
            "annotation_digest": analyzer._stable_json_sha256(
                [
                    {
                        "answer_id": row["answer_id"],
                        "annotation": row["annotation"],
                    }
                    for row in rows
                ]
            ),
            "critic_gate_digest": analyzer._stable_json_sha256(critic_gate),
        },
        output_json=metrics,
    )
    public_report = analyzer._build_public_report(
        profile=profile,
        technical=technical,
        technical_review=technical_review,
        metrics=metrics,
    )
    evidence, answer_corpus = await asyncio.gather(
        analyzer._evidence_sample(run_id),
        analyzer._full_answer_context(
            run_id,
            critic_gate=critic_gate,
            critic_rows=rows,
            expected_corpus_cells=expected_corpus_cells,
        ),
    )
    final, illustrations = await analyzer._run_report_branches(
        run_id,
        public_report=public_report,
        evidence=evidence,
        answer_corpus=answer_corpus,
        brand_name=str(profile.get("brand_name") or "бренд"),
    )
    markdown = analyzer._render_markdown(final)
    site_preview = await get_saved_site_preview(run_id)
    report_json = {
        **public_report,
        "narrative": {
            "headline": final.get("headline"),
            "headline_emphasis": final.get("headline_emphasis") or [],
            "verdict": final.get("verdict"),
            "executive_summary": final.get("executive_summary"),
            "actions": final.get("actions") or [],
        },
        "illustrations": illustrations,
        **({"site_preview": site_preview} if site_preview else {}),
    }
    reader_copy_manifest = await analyzer._save_reader_copy_manifest(
        run_id,
        final_report=final,
        public_report=public_report,
        illustrations=illustrations,
        analysis_markdown=markdown,
        report_json=report_json,
    )
    try:
        await replace_completed_publication(
            run_id=run_id,
            expected_snapshot_digest=expected_snapshot_digest,
            report_json=report_json,
            analysis_markdown=markdown,
            reader_copy_manifest=reader_copy_manifest,
        )
    except PublicationContractError as exc:
        raise RebuildGuardError(
            "Публичный отчёт изменился или перестал проходить контракт "
            "публикации во время пересборки; результат не опубликован."
        ) from exc

    return {
        "run_id": run_id,
        "answers_reused": len(rows),
        "control_pages_refreshed": refreshed,
        "technical_score": technical.get("score"),
        "parent_web": metrics.get("parent_discovery", {}).get("web"),
        "portfolio_web": metrics.get("portfolio_visibility", {}).get("web"),
        "illustrations": len(
            [item for item in illustrations if item.get("file_url")]
        ),
    }


def _args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Пересобрать отчёт из сохранённых аннотаций без вызова "
            "модельной панели и без повторной аннотации ответов."
        )
    )
    parser.add_argument("run_id", help="UUID завершённой проверки")
    parser.add_argument(
        "--refresh-control-pages",
        action="store_true",
        help="Повторно полностью прочитать только контрольные HTML-страницы.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _args(argv)
    try:
        result = asyncio.run(
            rebuild_from_saved_annotations(
                args.run_id,
                refresh_control_pages=args.refresh_control_pages,
            )
        )
    except RebuildGuardError as exc:
        print(f"Отказ: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
