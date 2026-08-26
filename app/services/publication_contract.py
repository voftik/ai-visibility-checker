"""Immutable publication receipts for public AIV reports.

The analysis pipeline writes the report fields and a content-addressed receipt
in one database transaction.  Public detail/share routes verify that receipt
before returning a completed report.  Pre-receipt reports are migrated once by
an explicit legacy baseline: the baseline proves the snapshot seen at migration
time, but deliberately does not pretend to prove how the old report was built.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Run, RunArtifact, RunStatus
from app.services.live_russian_policy import (
    LIVE_RUSSIAN_POLICY_MANIFEST,
    trusted_live_russian_policy_manifest,
)
from app.services.reader_copy_registry import (
    READER_COPY_REGISTRY_MANIFEST,
    trusted_reader_copy_registry_manifest,
)
from app.services.report_editor import (
    validate_archived_editorial_cache,
    validate_editorial_cache,
)
from app.services.site_preview import (
    site_preview_asset_receipt as build_site_preview_asset_receipt,
)

PUBLICATION_RECEIPT_VERSION = "aiv-report-publication-receipt-v3"
LEGACY_PUBLICATION_BASELINE_VERSION = "aiv-report-publication-legacy-baseline-v2"
PUBLICATION_RECEIPT_PREFIX = "report_publication_receipt_"
IMMUTABLE_READER_COPY_PREFIX = "immutable_reader_copy_manifest_"
IMMUTABLE_ILLUSTRATION_QA_PREFIX = "immutable_illustration_qa_"
READER_COPY_MANIFEST_VERSION = "aiv-reader-copy-manifest-v5"
# Keep old literals here when introducing a new current manifest version.  The
# current code-owned version is also trusted, but it never replaces retained
# literals during a normal version bump.
TRUSTED_READER_COPY_MANIFEST_VERSIONS = frozenset(
    {
        "aiv-reader-copy-manifest-v5",
        READER_COPY_MANIFEST_VERSION,
    }
)
EDITORIAL_CACHE_PROOF_VERSION = "aiv-editorial-cache-proof-v1"

STATIC_DIR = Path(__file__).resolve().parents[2] / "static"
GENERATED_DIR = STATIC_DIR / "generated"
_SAVED_ANSWERS_ONLY_MARKER_KEY = "_aiv_saved_answers_only_reprocess"
_SAVED_ANSWERS_ONLY_MARKER_VERSION = "aiv-saved-answers-only-v1"
_SAVED_ANSWERS_ONLY_MODE = "saved_answers_only"


class PublicationContractError(RuntimeError):
    """A completed report has no valid receipt for its current public bytes."""


def has_visible_publication_snapshot(run: Run) -> bool:
    """Keep the last sealed report public during saved-answer reprocessing."""

    if run.status == RunStatus.completed:
        return True
    if run.status not in {
        RunStatus.pending,
        RunStatus.crawling,
        RunStatus.analyzing,
    }:
        return False
    config = run.config_json if isinstance(run.config_json, dict) else {}
    marker = config.get(_SAVED_ANSWERS_ONLY_MARKER_KEY)
    if not isinstance(marker, dict):
        return False
    previous = marker.get("previous_terminal_state")
    raw_sha256 = str(marker.get("raw_answers_sha256") or "")
    return bool(
        marker.get("version") == _SAVED_ANSWERS_ONLY_MARKER_VERSION
        and marker.get("mode") == _SAVED_ANSWERS_ONLY_MODE
        and marker.get("run_id") == run.id
        and isinstance(previous, dict)
        and previous.get("status") == RunStatus.completed.value
        and len(raw_sha256) == 64
        and all(char in "0123456789abcdef" for char in raw_sha256)
    )


async def persist_immutable_illustration_qa_receipt(
    *,
    run_id: str,
    sequence: int,
    file_url: str,
    image_sha256: str,
    source_artifact_key: str,
    source_input: dict[str, Any],
    source_output: dict[str, Any],
    source_prompt_version: str,
) -> dict[str, str]:
    """Snapshot a mutable QA work artifact under a content-addressed key."""

    if source_input.get("image_sha256") != image_sha256:
        raise PublicationContractError(
            f"Illustration {sequence} QA input does not bind its image"
        )
    qa_core = {
        "input": source_input,
        "output": source_output,
        "prompt_version": source_prompt_version,
    }
    qa_receipt_sha256 = stable_json_sha256(qa_core)
    artifact_key = f"{IMMUTABLE_ILLUSTRATION_QA_PREFIX}{sequence}_{qa_receipt_sha256}"
    input_json = {
        "sequence": sequence,
        "file_url": file_url,
        "image_sha256": image_sha256,
        "source_artifact_key": source_artifact_key,
        "qa_receipt_sha256": qa_receipt_sha256,
    }
    output_json = {
        **qa_core,
        "source_artifact_key": source_artifact_key,
        "receipt_sha256": qa_receipt_sha256,
    }
    from app.db import SessionLocal

    async with SessionLocal() as session:

        def matches(value: RunArtifact | None) -> bool:
            return bool(
                value is not None
                and value.stage_key == "publication"
                and value.status == "completed"
                and value.model is None
                and value.prompt_version == source_prompt_version
                and value.input_json == input_json
                and value.output_json == output_json
            )

        existing = (
            await session.execute(
                select(RunArtifact).where(
                    RunArtifact.run_id == run_id,
                    RunArtifact.artifact_key == artifact_key,
                )
            )
        ).scalar_one_or_none()
        if existing is not None:
            if not matches(existing):
                raise PublicationContractError(
                    "Immutable illustration QA receipt conflicts with persisted state"
                )
        else:
            session.add(
                RunArtifact(
                    run_id=run_id,
                    stage_key="publication",
                    artifact_key=artifact_key,
                    status="completed",
                    model=None,
                    prompt_version=source_prompt_version,
                    input_json=input_json,
                    output_json=output_json,
                )
            )
            try:
                await session.commit()
            except IntegrityError:
                await session.rollback()
                existing = (
                    await session.execute(
                        select(RunArtifact).where(
                            RunArtifact.run_id == run_id,
                            RunArtifact.artifact_key == artifact_key,
                        )
                    )
                ).scalar_one_or_none()
                if not matches(existing):
                    raise PublicationContractError(
                        "Immutable illustration QA receipt could not be "
                        "persisted without conflict"
                    )
    return {
        "qa_artifact_key": artifact_key,
        "qa_source_artifact_key": source_artifact_key,
        "qa_receipt_sha256": qa_receipt_sha256,
    }


def stable_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def text_sha256(value: str | None) -> str:
    """Hash the exact nullable text value without conflating null and empty."""

    return stable_json_sha256(value)


def publication_snapshot(
    *,
    report_json: dict[str, Any] | None,
    analysis_markdown: str | None,
) -> dict[str, Any]:
    return {
        "report_json_sha256": stable_json_sha256(report_json),
        "analysis_markdown_sha256": text_sha256(analysis_markdown),
        "report_json_is_null": report_json is None,
        "analysis_markdown_is_null": analysis_markdown is None,
    }


def publication_snapshot_digest(snapshot: dict[str, Any]) -> str:
    return stable_json_sha256(snapshot)


def _site_preview_receipt_for_report(
    *,
    run_id: str,
    report_json: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(report_json, dict) or report_json.get("site_preview") is None:
        return None
    try:
        return build_site_preview_asset_receipt(
            run_id,
            report_json.get("site_preview"),
        )
    except (OSError, TypeError, ValueError) as exc:
        raise PublicationContractError(
            "Published site preview is missing or fails byte-level integrity"
        ) from exc


def _site_preview_receipt_reasons(
    *,
    run_id: str,
    report_json: dict[str, Any] | None,
    receipt: Any,
    receipt_sha256: Any,
) -> list[str]:
    has_preview = bool(
        isinstance(report_json, dict) and report_json.get("site_preview") is not None
    )
    if not has_preview:
        reasons: list[str] = []
        if receipt is not None:
            reasons.append("site_preview_receipt_without_public_preview")
        if receipt_sha256 is not None:
            reasons.append("site_preview_digest_without_public_preview")
        return reasons
    try:
        expected = _site_preview_receipt_for_report(
            run_id=run_id,
            report_json=report_json,
        )
    except PublicationContractError:
        return ["site_preview_asset_invalid"]
    if not isinstance(expected, dict):
        return ["site_preview_asset_invalid"]
    reasons = []
    if receipt != expected:
        reasons.append("site_preview_asset_receipt_mismatch")
    if receipt_sha256 != expected.get("receipt_sha256"):
        reasons.append("site_preview_asset_receipt_digest_mismatch")
    return reasons


def publication_receipt_key(snapshot: dict[str, Any]) -> str:
    return PUBLICATION_RECEIPT_PREFIX + publication_snapshot_digest(snapshot)


def _editorial_cache_proof_reasons(
    *,
    receipt_name: str,
    receipt: Any,
    canonical_policy_manifest: dict[str, str],
    historical_read: bool,
) -> list[str]:
    """Validate current staging inputs or an archived immutable proof."""

    if not isinstance(receipt, dict):
        return [f"reader_copy_{receipt_name}_receipt_missing"]
    if receipt.get("accepted") is not True:
        return [f"reader_copy_{receipt_name}_receipt_not_accepted"]
    proof = receipt.get("cache_proof")
    if not isinstance(proof, dict):
        return [f"reader_copy_{receipt_name}_cache_proof_missing"]
    reasons: list[str] = []
    if proof.get("version") != EDITORIAL_CACHE_PROOF_VERSION:
        reasons.append(f"reader_copy_{receipt_name}_cache_proof_version_stale")
    proof_core = {key: value for key, value in proof.items() if key != "proof_sha256"}
    if proof.get("proof_sha256") != stable_json_sha256(proof_core):
        reasons.append(f"reader_copy_{receipt_name}_cache_proof_digest_mismatch")
    source = proof.get("source")
    result = proof.get("result")
    audit = proof.get("audit")
    prose_paths = proof.get("prose_paths")
    protected_terms = proof.get("protected_terms")
    if (
        not isinstance(source, dict)
        or not isinstance(result, dict)
        or not isinstance(audit, dict)
    ):
        reasons.append(f"reader_copy_{receipt_name}_cache_proof_payload_invalid")
        return reasons
    if prose_paths is not None and (
        not isinstance(prose_paths, list)
        or any(not isinstance(value, str) for value in prose_paths)
    ):
        reasons.append(f"reader_copy_{receipt_name}_cache_proof_paths_invalid")
        return reasons
    if not isinstance(protected_terms, list) or any(
        not isinstance(value, str) for value in protected_terms
    ):
        reasons.append(f"reader_copy_{receipt_name}_cache_proof_terms_invalid")
        return reasons
    if proof.get("source_sha256") != stable_json_sha256(source):
        reasons.append(f"reader_copy_{receipt_name}_cache_source_mismatch")
    if proof.get("result_sha256") != stable_json_sha256(result):
        reasons.append(f"reader_copy_{receipt_name}_cache_result_mismatch")
    if proof.get("audit_sha256") != audit.get("audit_sha256"):
        reasons.append(f"reader_copy_{receipt_name}_cache_audit_mismatch")
    if receipt.get("cache_proof_sha256") != proof.get("proof_sha256"):
        reasons.append(f"reader_copy_{receipt_name}_cache_receipt_mismatch")
    if receipt.get("audit_sha256") != audit.get("audit_sha256"):
        reasons.append(f"reader_copy_{receipt_name}_audit_receipt_mismatch")
    if receipt.get("result_report_sha256") != stable_json_sha256(result):
        reasons.append(f"reader_copy_{receipt_name}_result_receipt_mismatch")
    try:
        validator = (
            validate_archived_editorial_cache
            if historical_read
            else validate_editorial_cache
        )
        cache_valid = validator(
            source,
            result,
            audit,
            prose_paths=prose_paths,
            protected_terms=protected_terms,
            canonical_policy_manifest=canonical_policy_manifest,
        )
    except (KeyError, TypeError, ValueError):
        cache_valid = False
    if not cache_valid:
        reasons.append(f"reader_copy_{receipt_name}_editorial_cache_invalid")
    if receipt.get("cache_revalidated") is not True:
        reasons.append(f"reader_copy_{receipt_name}_cache_not_revalidated")
    return reasons


def validate_reader_copy_manifest(
    *,
    run_id: str,
    report_json: dict[str, Any] | None,
    analysis_markdown: str | None,
    manifest: Any,
    require_current_policy: bool = True,
) -> list[str]:
    """Re-derive the self-seal and exact public-field bindings."""

    if not isinstance(manifest, dict):
        return ["reader_copy_manifest_missing"]
    reasons: list[str] = []
    canonical_policy = manifest.get("canonical_policy")
    copy_registry = manifest.get("code_owned_copy_registry")
    if require_current_policy:
        if manifest.get("version") != READER_COPY_MANIFEST_VERSION:
            reasons.append("reader_copy_manifest_version_stale")
        if canonical_policy != LIVE_RUSSIAN_POLICY_MANIFEST.as_dict():
            reasons.append("reader_copy_manifest_canonical_policy_stale")
        if copy_registry != READER_COPY_REGISTRY_MANIFEST.as_dict():
            reasons.append("reader_copy_manifest_code_owned_registry_stale")
    else:
        if manifest.get("version") not in TRUSTED_READER_COPY_MANIFEST_VERSIONS:
            reasons.append("reader_copy_manifest_version_untrusted")
        if trusted_live_russian_policy_manifest(canonical_policy) is None:
            reasons.append("reader_copy_manifest_canonical_policy_untrusted")
        if trusted_reader_copy_registry_manifest(copy_registry) is None:
            reasons.append("reader_copy_manifest_code_owned_registry_untrusted")
    core = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest.get("manifest_sha256") != stable_json_sha256(core):
        reasons.append("reader_copy_manifest_self_digest_mismatch")
    if manifest.get("decision") != "pass":
        reasons.append("reader_copy_manifest_not_passed")
    if manifest.get("blocking_reasons") != []:
        reasons.append("reader_copy_manifest_has_blockers")
    if manifest.get("quality_complete") is not True:
        reasons.append("reader_copy_manifest_quality_incomplete")
    lint = manifest.get("lint")
    if (
        not isinstance(lint, dict)
        or not isinstance(canonical_policy, dict)
        or lint.get("policy_version") != canonical_policy.get("version")
        or lint.get("policy_sha256") != canonical_policy.get("sha256")
        or lint.get("blocking") is not False
        or lint.get("issues") != []
        or lint.get("omitted_issue_count") != 0
    ):
        reasons.append("reader_copy_manifest_lint_policy_stale_or_incomplete")
    publication = manifest.get("publication_contract")
    if not isinstance(publication, dict):
        reasons.append("reader_copy_publication_contract_missing")
    else:
        checks = publication.get("checks")
        if (
            not isinstance(checks, dict)
            or not checks
            or not all(value is True for value in checks.values())
        ):
            reasons.append("reader_copy_publication_checks_incomplete")
        if publication.get("blocking_reasons") != []:
            reasons.append("reader_copy_publication_contract_blocked")
        if publication.get("report_json_sha256") != stable_json_sha256(report_json):
            reasons.append("reader_copy_report_json_binding_mismatch")
        markdown_sha256 = (
            hashlib.sha256(analysis_markdown.encode("utf-8")).hexdigest()
            if isinstance(analysis_markdown, str)
            else None
        )
        if publication.get("analysis_markdown_sha256") != markdown_sha256:
            reasons.append("reader_copy_markdown_binding_mismatch")
    asset_receipts = manifest.get("illustration_asset_receipts")
    if not isinstance(asset_receipts, list):
        reasons.append("reader_copy_asset_receipts_missing")
    if manifest.get("illustration_asset_receipts_sha256") != stable_json_sha256(
        asset_receipts
    ):
        reasons.append("reader_copy_asset_receipts_digest_mismatch")
    reasons.extend(
        _site_preview_receipt_reasons(
            run_id=run_id,
            report_json=report_json,
            receipt=manifest.get("site_preview_asset_receipt"),
            receipt_sha256=manifest.get("site_preview_asset_receipt_sha256"),
        )
    )
    public_rows = _public_illustration_rows(report_json)
    editorial_receipts = manifest.get("editorial_receipts")
    if not isinstance(editorial_receipts, dict):
        reasons.append("reader_copy_editorial_receipts_missing")
        editorial_receipts = {}
    for receipt_name in ("final_report", "technical_review"):
        reasons.extend(
            _editorial_cache_proof_reasons(
                receipt_name=receipt_name,
                receipt=editorial_receipts.get(receipt_name),
                canonical_policy_manifest=(
                    canonical_policy if isinstance(canonical_policy, dict) else {}
                ),
                historical_read=not require_current_policy,
            )
        )
    illustration_copy_receipt = (
        editorial_receipts.get("illustrations")
        if isinstance(editorial_receipts, dict)
        else None
    )
    illustration_asset_receipt = (
        editorial_receipts.get("illustration_assets")
        if isinstance(editorial_receipts, dict)
        else None
    )
    if not isinstance(illustration_copy_receipt, dict):
        reasons.append("reader_copy_illustration_receipt_missing")
    elif illustration_copy_receipt.get(
        "accepted"
    ) is not True or illustration_copy_receipt.get("published_count") != len(
        public_rows
    ):
        reasons.append("reader_copy_illustration_receipt_mismatch")
    elif public_rows:
        reasons.extend(
            _editorial_cache_proof_reasons(
                receipt_name="illustrations",
                receipt=illustration_copy_receipt,
                canonical_policy_manifest=(
                    canonical_policy if isinstance(canonical_policy, dict) else {}
                ),
                historical_read=not require_current_policy,
            )
        )
    elif illustration_copy_receipt.get("state") != "not_published":
        reasons.append("reader_copy_zero_illustration_receipt_state_invalid")
    if not isinstance(illustration_asset_receipt, dict):
        reasons.append("reader_copy_illustration_asset_receipt_missing")
    elif (
        illustration_asset_receipt.get("accepted") is not True
        or illustration_asset_receipt.get("published_count") != len(public_rows)
        or illustration_asset_receipt.get("verified_count") != len(public_rows)
    ):
        reasons.append("reader_copy_illustration_asset_receipt_mismatch")
    policy = (
        illustration_asset_receipt.get("publication_policy")
        if isinstance(illustration_asset_receipt, dict)
        else None
    )
    if (
        not isinstance(policy, dict)
        or policy.get("publish_only_verified_assets") is not True
    ):
        reasons.append("reader_copy_illustration_policy_missing")
    elif not public_rows and policy.get("zero_assets_allowed") is not True:
        reasons.append("reader_copy_zero_illustration_policy_missing")
    elif public_rows and policy.get("verified_subset_allowed") is not True:
        reasons.append("reader_copy_subset_illustration_policy_missing")
    return list(dict.fromkeys(reasons))


async def _stage_immutable_reader_copy_manifest(
    session: AsyncSession,
    *,
    run_id: str,
    report_json: dict[str, Any],
    analysis_markdown: str,
    manifest: dict[str, Any],
) -> tuple[str, str]:
    reasons = validate_reader_copy_manifest(
        run_id=run_id,
        report_json=report_json,
        analysis_markdown=analysis_markdown,
        manifest=manifest,
    )
    if reasons:
        raise PublicationContractError(
            "Reader-copy manifest cannot be published: " + ", ".join(reasons)
        )
    manifest_sha256 = str(manifest["manifest_sha256"])
    artifact_key = IMMUTABLE_READER_COPY_PREFIX + manifest_sha256
    expected_input = {
        "report_json_sha256": stable_json_sha256(report_json),
        "analysis_markdown_sha256": hashlib.sha256(
            analysis_markdown.encode("utf-8")
        ).hexdigest(),
        "site_preview_asset_receipt_sha256": manifest.get(
            "site_preview_asset_receipt_sha256"
        ),
    }
    existing = (
        await session.execute(
            select(RunArtifact).where(
                RunArtifact.run_id == run_id,
                RunArtifact.artifact_key == artifact_key,
            )
        )
    ).scalar_one_or_none()
    if existing is not None:
        if (
            existing.status != "completed"
            or existing.stage_key != "publication"
            or existing.model is not None
            or existing.prompt_version != str(manifest.get("version") or "")
            or existing.input_json != expected_input
            or existing.output_json != manifest
        ):
            raise PublicationContractError(
                "Immutable reader-copy manifest conflicts with persisted state"
            )
        return artifact_key, manifest_sha256
    session.add(
        RunArtifact(
            run_id=run_id,
            stage_key="publication",
            artifact_key=artifact_key,
            status="completed",
            model=None,
            prompt_version=str(manifest.get("version") or ""),
            input_json=expected_input,
            output_json=manifest,
        )
    )
    return artifact_key, manifest_sha256


def resolve_illustration_path(run_id: str, file_url: Any) -> Path | None:
    """Resolve only a generated file physically owned by ``run_id``."""

    if not isinstance(file_url, str) or not file_url.strip():
        return None
    parsed = urlparse(file_url.strip())
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        return None
    safe_run_id = re.sub(r"[^a-zA-Z0-9_-]", "", run_id)
    expected_prefix = f"/static/generated/{safe_run_id}/"
    if not safe_run_id or not parsed.path.startswith(expected_prefix):
        return None
    relative = parsed.path.removeprefix("/static/")
    candidate = (STATIC_DIR / relative).resolve()
    run_root = (GENERATED_DIR / safe_run_id).resolve()
    if not candidate.is_relative_to(run_root) or not candidate.is_file():
        return None
    return candidate


def _public_illustration_rows(
    report_json: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not isinstance(report_json, dict):
        return []
    rows = report_json.get("illustrations")
    if not isinstance(rows, list):
        return []
    # Older reports kept failed illustration attempts as copy-only rows with
    # ``file_url = null``.  They are not published assets and therefore do not
    # require a file receipt.  Current reports omit them altogether.
    return [
        row
        for row in rows
        if isinstance(row, dict)
        and isinstance(row.get("file_url"), str)
        and bool(row["file_url"])
    ]


def _validate_asset_receipts(
    *,
    run_id: str,
    report_json: dict[str, Any] | None,
    receipts: Any,
    legacy_baseline: bool,
) -> list[str]:
    """Verify URL, ownership, existence and bytes for every published asset."""

    rows = _public_illustration_rows(report_json)
    reasons: list[str] = []
    if not isinstance(receipts, list):
        return ["illustration_asset_receipts_missing"] if rows else []
    valid_receipts = [
        receipt
        for receipt in receipts
        if isinstance(receipt, dict)
        and isinstance(receipt.get("sequence"), int)
        and not isinstance(receipt.get("sequence"), bool)
    ]
    by_sequence = {receipt.get("sequence"): receipt for receipt in valid_receipts}
    if len(valid_receipts) != len(receipts) or len(by_sequence) != len(receipts):
        reasons.append("illustration_asset_receipts_invalid_or_duplicate")
    seen_sequences: set[int] = set()
    for position, row in enumerate(rows, start=1):
        sequence = row.get("sequence")
        if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 1:
            reasons.append("published_illustration_sequence_invalid")
            continue
        if sequence in seen_sequences:
            reasons.append("published_illustration_sequence_duplicate")
            continue
        seen_sequences.add(sequence)
        receipt = by_sequence.get(sequence)
        if not isinstance(receipt, dict):
            reasons.append(f"illustration_{sequence}_asset_receipt_missing")
            continue
        file_url = row.get("file_url")
        if not isinstance(file_url, str) or not file_url:
            reasons.append(f"illustration_{sequence}_file_url_missing")
            continue
        if receipt.get("file_url") != file_url:
            reasons.append(f"illustration_{sequence}_file_url_mismatch")
        path = resolve_illustration_path(run_id, file_url)
        if path is None:
            reasons.append(f"illustration_{sequence}_file_missing_or_foreign")
            continue
        actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if receipt.get("image_sha256") != actual_sha256:
            reasons.append(f"illustration_{sequence}_content_sha256_mismatch")
        if receipt.get("image_bytes") != path.stat().st_size and not legacy_baseline:
            reasons.append(f"illustration_{sequence}_content_size_mismatch")
        if legacy_baseline:
            if receipt.get("baseline_integrity_verified") is not True:
                reasons.append(f"illustration_{sequence}_legacy_integrity_not_verified")
            if receipt.get("historical_qa_provenance") is not False:
                reasons.append(f"illustration_{sequence}_legacy_provenance_ambiguous")
        else:
            if receipt.get("qa_verified") is not True:
                reasons.append(f"illustration_{sequence}_qa_not_verified")
            expected_filename = (
                f"{sequence:02d}-{actual_sha256}{path.suffix.casefold()}"
            )
            if (
                receipt.get("content_addressed_filename") is not True
                or path.name != expected_filename
            ):
                reasons.append(
                    f"illustration_{sequence}_filename_not_content_addressed"
                )
            qa_receipt_sha256 = receipt.get("qa_receipt_sha256")
            expected_qa_key = (
                f"{IMMUTABLE_ILLUSTRATION_QA_PREFIX}{sequence}_{qa_receipt_sha256}"
            )
            if receipt.get("qa_artifact_key") != expected_qa_key:
                reasons.append(f"illustration_{sequence}_qa_binding_invalid")
            if not isinstance(qa_receipt_sha256, str) or len(qa_receipt_sha256) != 64:
                reasons.append(f"illustration_{sequence}_qa_digest_missing")
            if not isinstance(receipt.get("qa_source_artifact_key"), str):
                reasons.append(f"illustration_{sequence}_qa_source_missing")
    if set(by_sequence) != seen_sequences:
        reasons.append("illustration_asset_receipt_sequence_mismatch")
    return list(dict.fromkeys(reasons))


async def _validate_qa_artifact_bindings(
    session: AsyncSession,
    *,
    run_id: str,
    receipts: Any,
    legacy_baseline: bool,
) -> list[str]:
    """Recompute every same-run QA receipt referenced by public assets."""

    if legacy_baseline:
        return []
    if not isinstance(receipts, list):
        return ["illustration_asset_receipts_missing"]
    reasons: list[str] = []
    for receipt in receipts:
        if not isinstance(receipt, dict):
            continue
        sequence = receipt.get("sequence")
        artifact_key = receipt.get("qa_artifact_key")
        if not isinstance(sequence, int) or isinstance(sequence, bool):
            continue
        if not isinstance(artifact_key, str) or not artifact_key:
            reasons.append(f"illustration_{sequence}_qa_artifact_missing")
            continue
        artifact = (
            await session.execute(
                select(RunArtifact).where(
                    RunArtifact.run_id == run_id,
                    RunArtifact.artifact_key == artifact_key,
                )
            )
        ).scalar_one_or_none()
        if (
            artifact is None
            or artifact.status != "completed"
            or not isinstance(artifact.input_json, dict)
            or not isinstance(artifact.output_json, dict)
        ):
            reasons.append(f"illustration_{sequence}_qa_artifact_invalid")
            continue
        envelope = artifact.output_json
        qa_input = envelope.get("input")
        qa_output = envelope.get("output")
        qa_prompt_version = envelope.get("prompt_version")
        source_artifact_key = envelope.get("source_artifact_key")
        if not isinstance(qa_input, dict) or not isinstance(qa_output, dict):
            reasons.append(f"illustration_{sequence}_qa_envelope_invalid")
            continue
        qa_core = {
            "input": qa_input,
            "output": qa_output,
            "prompt_version": qa_prompt_version,
        }
        actual_digest = stable_json_sha256(qa_core)
        expected_key = f"{IMMUTABLE_ILLUSTRATION_QA_PREFIX}{sequence}_{actual_digest}"
        expected_input = {
            "sequence": sequence,
            "file_url": receipt.get("file_url"),
            "image_sha256": receipt.get("image_sha256"),
            "source_artifact_key": receipt.get("qa_source_artifact_key"),
            "qa_receipt_sha256": actual_digest,
        }
        if (
            artifact.artifact_key != expected_key
            or artifact.stage_key != "publication"
            or artifact.model is not None
            or artifact.prompt_version != qa_prompt_version
            or artifact.input_json != expected_input
        ):
            reasons.append(f"illustration_{sequence}_qa_envelope_mismatch")
        if qa_input.get("image_sha256") != receipt.get("image_sha256"):
            reasons.append(f"illustration_{sequence}_qa_image_mismatch")
        if (
            receipt.get("qa_receipt_sha256") != actual_digest
            or envelope.get("receipt_sha256") != actual_digest
        ):
            reasons.append(f"illustration_{sequence}_qa_receipt_mismatch")
        if receipt.get("qa_source_artifact_key") != source_artifact_key:
            reasons.append(f"illustration_{sequence}_qa_source_mismatch")
    return list(dict.fromkeys(reasons))


def build_publication_receipt(
    *,
    run_id: str,
    report_json: dict[str, Any] | None,
    analysis_markdown: str | None,
    reader_copy_manifest_artifact_key: str | None,
    reader_copy_manifest_sha256: str | None,
    illustration_asset_receipts: list[dict[str, Any]],
    site_preview_asset_receipt: dict[str, Any] | None = None,
    legacy_baseline: bool = False,
) -> dict[str, Any]:
    snapshot = publication_snapshot(
        report_json=report_json,
        analysis_markdown=analysis_markdown,
    )
    core = {
        "version": (
            LEGACY_PUBLICATION_BASELINE_VERSION
            if legacy_baseline
            else PUBLICATION_RECEIPT_VERSION
        ),
        "run_id": run_id,
        "snapshot": snapshot,
        "snapshot_digest": publication_snapshot_digest(snapshot),
        "reader_copy_manifest_artifact_key": (reader_copy_manifest_artifact_key),
        "reader_copy_manifest_sha256": reader_copy_manifest_sha256,
        "illustration_asset_receipts": illustration_asset_receipts,
        "illustration_asset_receipts_sha256": stable_json_sha256(
            illustration_asset_receipts
        ),
        "site_preview_asset_receipt": site_preview_asset_receipt,
        "site_preview_asset_receipt_sha256": (
            site_preview_asset_receipt.get("receipt_sha256")
            if isinstance(site_preview_asset_receipt, dict)
            else None
        ),
        "legacy_baseline": legacy_baseline,
        "historical_pipeline_provenance": not legacy_baseline,
    }
    return {**core, "receipt_sha256": stable_json_sha256(core)}


def validate_publication_receipt(
    *,
    run_id: str,
    report_json: dict[str, Any] | None,
    analysis_markdown: str | None,
    receipt: Any,
) -> list[str]:
    reasons: list[str] = []
    if not isinstance(receipt, dict):
        return ["publication_receipt_missing"]
    legacy = receipt.get("legacy_baseline") is True
    expected_version = (
        LEGACY_PUBLICATION_BASELINE_VERSION if legacy else PUBLICATION_RECEIPT_VERSION
    )
    if receipt.get("version") != expected_version:
        reasons.append("publication_receipt_version_mismatch")
    if receipt.get("historical_pipeline_provenance") is not (not legacy):
        reasons.append("publication_provenance_flag_mismatch")
    if receipt.get("run_id") != run_id:
        reasons.append("publication_receipt_run_mismatch")
    snapshot = publication_snapshot(
        report_json=report_json,
        analysis_markdown=analysis_markdown,
    )
    if receipt.get("snapshot") != snapshot:
        reasons.append("publication_snapshot_mismatch")
    if receipt.get("snapshot_digest") != publication_snapshot_digest(snapshot):
        reasons.append("publication_snapshot_digest_mismatch")
    asset_receipts = receipt.get("illustration_asset_receipts")
    if receipt.get("illustration_asset_receipts_sha256") != stable_json_sha256(
        asset_receipts
    ):
        reasons.append("illustration_asset_receipts_digest_mismatch")
    reasons.extend(
        _validate_asset_receipts(
            run_id=run_id,
            report_json=report_json,
            receipts=asset_receipts,
            legacy_baseline=legacy,
        )
    )
    reasons.extend(
        _site_preview_receipt_reasons(
            run_id=run_id,
            report_json=report_json,
            receipt=receipt.get("site_preview_asset_receipt"),
            receipt_sha256=receipt.get("site_preview_asset_receipt_sha256"),
        )
    )
    core = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if receipt.get("receipt_sha256") != stable_json_sha256(core):
        reasons.append("publication_receipt_digest_mismatch")
    if not legacy:
        manifest_sha256 = receipt.get("reader_copy_manifest_sha256")
        manifest_key = receipt.get("reader_copy_manifest_artifact_key")
        if not isinstance(manifest_sha256, str) or not manifest_sha256:
            reasons.append("reader_copy_manifest_digest_binding_missing")
        if not isinstance(
            manifest_key, str
        ) or manifest_key != IMMUTABLE_READER_COPY_PREFIX + str(manifest_sha256):
            reasons.append("reader_copy_manifest_artifact_binding_missing")
    elif (
        receipt.get("reader_copy_manifest_artifact_key") is not None
        or receipt.get("reader_copy_manifest_sha256") is not None
    ):
        reasons.append("legacy_reader_copy_provenance_must_be_unknown")
    return list(dict.fromkeys(reasons))


def _publication_artifact_envelope_reasons(
    artifact: RunArtifact,
    receipt: dict[str, Any],
) -> list[str]:
    """Validate the immutable DB envelope around a self-sealed receipt."""

    reasons: list[str] = []
    legacy = receipt.get("legacy_baseline") is True
    expected_key = (
        publication_receipt_key(receipt.get("snapshot") or {})
        if legacy
        else PUBLICATION_RECEIPT_PREFIX + str(receipt.get("receipt_sha256") or "")
    )
    expected_input = (
        {
            "migration": "explicit_completed_snapshot_baseline",
            "snapshot_digest": receipt.get("snapshot_digest"),
            "historical_pipeline_provenance": False,
            "site_preview_asset_receipt_sha256": receipt.get(
                "site_preview_asset_receipt_sha256"
            ),
        }
        if legacy
        else {
            "snapshot_digest": receipt.get("snapshot_digest"),
            "reader_copy_manifest_artifact_key": receipt.get(
                "reader_copy_manifest_artifact_key"
            ),
            "reader_copy_manifest_sha256": receipt.get("reader_copy_manifest_sha256"),
            "site_preview_asset_receipt_sha256": receipt.get(
                "site_preview_asset_receipt_sha256"
            ),
        }
    )
    if artifact.artifact_key != expected_key:
        reasons.append("publication_artifact_key_mismatch")
    if artifact.stage_key != "publication":
        reasons.append("publication_artifact_stage_mismatch")
    if artifact.status != "completed":
        reasons.append("publication_artifact_not_completed")
    if artifact.model is not None:
        reasons.append("publication_artifact_model_mismatch")
    if artifact.prompt_version != receipt.get("version"):
        reasons.append("publication_artifact_version_mismatch")
    if artifact.input_json != expected_input:
        reasons.append("publication_artifact_input_mismatch")
    if artifact.output_json != receipt:
        reasons.append("publication_artifact_output_mismatch")
    return reasons


async def stage_publication_receipt(
    session: AsyncSession,
    *,
    run_id: str,
    report_json: dict[str, Any],
    analysis_markdown: str,
    reader_copy_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Stage an immutable receipt in the caller's publication transaction."""

    (
        reader_copy_manifest_artifact_key,
        reader_copy_manifest_sha256,
    ) = await _stage_immutable_reader_copy_manifest(
        session,
        run_id=run_id,
        report_json=report_json,
        analysis_markdown=analysis_markdown,
        manifest=reader_copy_manifest,
    )
    illustration_asset_receipts = reader_copy_manifest.get(
        "illustration_asset_receipts"
    )
    if not isinstance(illustration_asset_receipts, list):
        raise PublicationContractError(
            "Reader-copy manifest has no illustration asset receipts"
        )
    qa_reasons = await _validate_qa_artifact_bindings(
        session,
        run_id=run_id,
        receipts=illustration_asset_receipts,
        legacy_baseline=False,
    )
    if qa_reasons:
        raise PublicationContractError(
            "Publication QA receipts are invalid: " + ", ".join(qa_reasons)
        )
    site_preview_receipt = reader_copy_manifest.get("site_preview_asset_receipt")
    receipt = build_publication_receipt(
        run_id=run_id,
        report_json=report_json,
        analysis_markdown=analysis_markdown,
        reader_copy_manifest_artifact_key=(reader_copy_manifest_artifact_key),
        reader_copy_manifest_sha256=reader_copy_manifest_sha256,
        illustration_asset_receipts=illustration_asset_receipts,
        site_preview_asset_receipt=(
            site_preview_receipt if isinstance(site_preview_receipt, dict) else None
        ),
    )
    reasons = validate_publication_receipt(
        run_id=run_id,
        report_json=report_json,
        analysis_markdown=analysis_markdown,
        receipt=receipt,
    )
    if reasons:
        raise PublicationContractError(
            "Publication receipt cannot be staged: " + ", ".join(reasons)
        )
    # Include the editorial/asset binding in the immutable identity.  The same
    # report bytes may be re-audited by a newer policy without overwriting the
    # earlier receipt.
    key = PUBLICATION_RECEIPT_PREFIX + str(receipt["receipt_sha256"])
    existing = (
        await session.execute(
            select(RunArtifact).where(
                RunArtifact.run_id == run_id,
                RunArtifact.artifact_key == key,
            )
        )
    ).scalar_one_or_none()
    if existing is not None:
        if _publication_artifact_envelope_reasons(existing, receipt):
            raise PublicationContractError(
                "Immutable publication receipt conflicts with persisted state"
            )
        return receipt
    session.add(
        RunArtifact(
            run_id=run_id,
            stage_key="publication",
            artifact_key=key,
            status="completed",
            model=None,
            prompt_version=PUBLICATION_RECEIPT_VERSION,
            input_json={
                "snapshot_digest": receipt["snapshot_digest"],
                "reader_copy_manifest_artifact_key": (
                    reader_copy_manifest_artifact_key
                ),
                "reader_copy_manifest_sha256": reader_copy_manifest_sha256,
                "site_preview_asset_receipt_sha256": receipt.get(
                    "site_preview_asset_receipt_sha256"
                ),
            },
            output_json=receipt,
        )
    )
    return receipt


async def _validate_receipt_dependencies(
    session: AsyncSession,
    *,
    run_id: str,
    report_json: dict[str, Any] | None,
    analysis_markdown: str | None,
    receipt: dict[str, Any],
) -> list[str]:
    """Reload every immutable dependency named by one publication receipt."""

    legacy = receipt.get("legacy_baseline") is True
    asset_receipts = receipt.get("illustration_asset_receipts")
    reasons = await _validate_qa_artifact_bindings(
        session,
        run_id=run_id,
        receipts=asset_receipts,
        legacy_baseline=legacy,
    )
    if legacy:
        return reasons
    artifact_key = receipt.get("reader_copy_manifest_artifact_key")
    manifest_sha256 = receipt.get("reader_copy_manifest_sha256")
    if not isinstance(artifact_key, str) or not isinstance(manifest_sha256, str):
        return [*reasons, "reader_copy_manifest_binding_missing"]
    artifact = (
        await session.execute(
            select(RunArtifact).where(
                RunArtifact.run_id == run_id,
                RunArtifact.artifact_key == artifact_key,
            )
        )
    ).scalar_one_or_none()
    if (
        artifact is None
        or artifact.status != "completed"
        or not isinstance(artifact.output_json, dict)
    ):
        return [*reasons, "reader_copy_manifest_artifact_invalid"]
    manifest = artifact.output_json
    if artifact.stage_key != "publication" or artifact.model is not None:
        reasons.append("reader_copy_manifest_artifact_envelope_mismatch")
    if artifact.prompt_version != str(manifest.get("version") or ""):
        reasons.append("reader_copy_manifest_prompt_version_mismatch")
    if artifact_key != IMMUTABLE_READER_COPY_PREFIX + manifest_sha256:
        reasons.append("reader_copy_manifest_key_mismatch")
    if manifest.get("manifest_sha256") != manifest_sha256:
        reasons.append("reader_copy_manifest_receipt_digest_mismatch")
    expected_input = {
        "report_json_sha256": stable_json_sha256(report_json),
        "analysis_markdown_sha256": (
            hashlib.sha256(analysis_markdown.encode("utf-8")).hexdigest()
            if isinstance(analysis_markdown, str)
            else None
        ),
        "site_preview_asset_receipt_sha256": manifest.get(
            "site_preview_asset_receipt_sha256"
        ),
    }
    if artifact.input_json != expected_input:
        reasons.append("reader_copy_manifest_artifact_input_mismatch")
    reasons.extend(
        validate_reader_copy_manifest(
            run_id=run_id,
            report_json=report_json,
            analysis_markdown=analysis_markdown,
            manifest=manifest,
            require_current_policy=False,
        )
    )
    if manifest.get("illustration_asset_receipts") != asset_receipts:
        reasons.append("reader_copy_manifest_asset_receipts_mismatch")
    if manifest.get("site_preview_asset_receipt") != receipt.get(
        "site_preview_asset_receipt"
    ):
        reasons.append("reader_copy_manifest_site_preview_receipt_mismatch")
    return list(dict.fromkeys(reasons))


def _legacy_asset_receipts(
    *,
    run_id: str,
    report_json: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for position, row in enumerate(_public_illustration_rows(report_json), start=1):
        sequence = row.get("sequence")
        if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 1:
            raise PublicationContractError(
                f"Legacy illustration {position} has no valid sequence"
            )
        file_url = row.get("file_url")
        path = resolve_illustration_path(run_id, file_url)
        if path is None:
            raise PublicationContractError(
                f"Legacy illustration {sequence} is missing or outside run scope"
            )
        receipts.append(
            {
                "sequence": sequence,
                "file_url": file_url,
                "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "qa_verified": False,
                "baseline_integrity_verified": True,
                "verification_source": "legacy_read_baseline",
                "historical_qa_provenance": False,
            }
        )
    return receipts


async def ensure_publication_contract(
    session: AsyncSession,
    run: Run,
    *,
    allow_legacy_baseline: bool = True,
) -> dict[str, Any] | None:
    """Verify a completed report, creating one explicit legacy baseline only."""

    if not has_visible_publication_snapshot(run):
        return None
    # Keep the exact public values outside ORM state before the optional legacy
    # baseline commit.  A concurrent first-read migration can lose the UNIQUE
    # race and force ``session.rollback()``; SQLAlchemy expires ORM instances on
    # rollback even when ``expire_on_commit=False``.  Reusing these plain values
    # avoids an implicit async refresh (and a possible MissingGreenlet) while
    # still validating the same snapshot that entered this function.
    run_id = str(run.id)
    report_json = run.report_json
    analysis_markdown = run.analysis_markdown
    snapshot = publication_snapshot(
        report_json=report_json,
        analysis_markdown=analysis_markdown,
    )
    artifacts = list(
        (
            await session.execute(
                select(RunArtifact).where(
                    RunArtifact.run_id == run_id,
                    RunArtifact.artifact_key.like(f"{PUBLICATION_RECEIPT_PREFIX}%"),
                )
            )
        ).scalars()
    )
    exact: RunArtifact | None = None
    for item in sorted(artifacts, key=lambda value: value.id, reverse=True):
        if item.status != "completed" or not isinstance(item.output_json, dict):
            continue
        receipt_reasons = validate_publication_receipt(
            run_id=run_id,
            report_json=report_json,
            analysis_markdown=analysis_markdown,
            receipt=item.output_json,
        )
        if receipt_reasons:
            continue
        if _publication_artifact_envelope_reasons(item, item.output_json):
            continue
        dependency_reasons = await _validate_receipt_dependencies(
            session,
            run_id=run_id,
            report_json=report_json,
            analysis_markdown=analysis_markdown,
            receipt=item.output_json,
        )
        if not dependency_reasons:
            exact = item
            break
    if exact is None:
        if artifacts:
            raise PublicationContractError(
                "Completed report bytes do not match any publication receipt"
            )
        if not allow_legacy_baseline:
            raise PublicationContractError(
                "Completed legacy report has no publication receipt"
            )
        receipts = _legacy_asset_receipts(
            run_id=run_id,
            report_json=report_json,
        )
        site_preview_receipt = _site_preview_receipt_for_report(
            run_id=run_id,
            report_json=report_json,
        )
        baseline = build_publication_receipt(
            run_id=run_id,
            report_json=report_json,
            analysis_markdown=analysis_markdown,
            reader_copy_manifest_artifact_key=None,
            reader_copy_manifest_sha256=None,
            illustration_asset_receipts=receipts,
            site_preview_asset_receipt=site_preview_receipt,
            legacy_baseline=True,
        )
        key = publication_receipt_key(snapshot)
        exact = RunArtifact(
            run_id=run_id,
            stage_key="publication",
            artifact_key=key,
            status="completed",
            model=None,
            prompt_version=LEGACY_PUBLICATION_BASELINE_VERSION,
            input_json={
                "migration": "explicit_completed_snapshot_baseline",
                "snapshot_digest": baseline["snapshot_digest"],
                "historical_pipeline_provenance": False,
                "site_preview_asset_receipt_sha256": baseline.get(
                    "site_preview_asset_receipt_sha256"
                ),
            },
            output_json=baseline,
        )
        session.add(exact)
        try:
            await session.commit()
        except IntegrityError:
            # Two first reads can race during rollout.  The unique
            # (run_id, artifact_key) index elects one identical baseline;
            # reload it and validate its bytes below.
            await session.rollback()
            exact = (
                await session.execute(
                    select(RunArtifact).where(
                        RunArtifact.run_id == run_id,
                        RunArtifact.artifact_key == key,
                    )
                )
            ).scalar_one_or_none()
            if exact is None:
                raise PublicationContractError(
                    "Legacy publication baseline could not be persisted"
                )
    if exact.status != "completed":
        raise PublicationContractError("Publication receipt is not completed")
    receipt = exact.output_json
    reasons = validate_publication_receipt(
        run_id=run_id,
        report_json=report_json,
        analysis_markdown=analysis_markdown,
        receipt=receipt,
    )
    if reasons:
        raise PublicationContractError(
            "Completed report failed publication verification: " + ", ".join(reasons)
        )
    envelope_reasons = _publication_artifact_envelope_reasons(exact, receipt)
    if envelope_reasons:
        raise PublicationContractError(
            "Completed report receipt envelope failed verification: "
            + ", ".join(envelope_reasons)
        )
    dependency_reasons = await _validate_receipt_dependencies(
        session,
        run_id=run_id,
        report_json=report_json,
        analysis_markdown=analysis_markdown,
        receipt=receipt,
    )
    if dependency_reasons:
        raise PublicationContractError(
            "Completed report dependencies failed publication verification: "
            + ", ".join(dependency_reasons)
        )
    return receipt


async def replace_completed_publication(
    *,
    run_id: str,
    expected_snapshot_digest: str,
    report_json: dict[str, Any],
    analysis_markdown: str,
    reader_copy_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Atomically CAS-replace one already verified non-legacy publication.

    Operator tools use this entrypoint instead of assigning public fields
    directly.  The current receipt, new immutable manifest, new receipt and
    both public fields are verified and committed under one SQLite write lock.
    """

    if not isinstance(report_json, dict) or not isinstance(analysis_markdown, str):
        raise PublicationContractError(
            "Replacement publication fields have invalid types"
        )
    from app.db import SessionLocal

    async with SessionLocal() as session:
        try:
            await session.execute(text("BEGIN IMMEDIATE"))
            run = await session.get(Run, run_id)
            if run is None:
                raise PublicationContractError("Run not found")
            if run.status != RunStatus.completed:
                raise PublicationContractError(
                    "Only a completed publication can be replaced"
                )
            current_receipt = await ensure_publication_contract(
                session,
                run,
                allow_legacy_baseline=False,
            )
            if (
                not isinstance(current_receipt, dict)
                or current_receipt.get("legacy_baseline") is True
            ):
                raise PublicationContractError(
                    "Legacy publications require the full saved-data reprocess"
                )
            current_snapshot = publication_snapshot(
                report_json=run.report_json,
                analysis_markdown=run.analysis_markdown,
            )
            if (
                publication_snapshot_digest(current_snapshot)
                != expected_snapshot_digest
            ):
                raise PublicationContractError(
                    "Publication changed while the replacement was prepared"
                )
            receipt = await stage_publication_receipt(
                session,
                run_id=run_id,
                report_json=report_json,
                analysis_markdown=analysis_markdown,
                reader_copy_manifest=reader_copy_manifest,
            )
            run.report_json = report_json
            run.analysis_markdown = analysis_markdown
            await session.commit()
            return receipt
        except Exception:
            await session.rollback()
            raise


__all__ = [
    "EDITORIAL_CACHE_PROOF_VERSION",
    "GENERATED_DIR",
    "IMMUTABLE_ILLUSTRATION_QA_PREFIX",
    "IMMUTABLE_READER_COPY_PREFIX",
    "LEGACY_PUBLICATION_BASELINE_VERSION",
    "PUBLICATION_RECEIPT_PREFIX",
    "PUBLICATION_RECEIPT_VERSION",
    "READER_COPY_MANIFEST_VERSION",
    "PublicationContractError",
    "build_publication_receipt",
    "ensure_publication_contract",
    "has_visible_publication_snapshot",
    "persist_immutable_illustration_qa_receipt",
    "publication_receipt_key",
    "publication_snapshot",
    "publication_snapshot_digest",
    "replace_completed_publication",
    "resolve_illustration_path",
    "stable_json_sha256",
    "stage_publication_receipt",
    "validate_publication_receipt",
    "validate_reader_copy_manifest",
]
