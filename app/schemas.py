from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.models import RunStatus
from app.services.publication_contract import has_visible_publication_snapshot


class CreateRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    domain: str = Field(min_length=1, max_length=2048)


class CreateRunResponse(BaseModel):
    run_id: str


class RunSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    created_at: datetime
    status: RunStatus
    domain: str | None
    progress_percent: int
    stage_key: str | None
    stage_label: str | None
    run_state: str
    state_revision: int
    queue_position: int | None = None
    queue_total: int | None = None
    started_at: datetime | None
    finished_at: datetime | None


class RunLookupRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ids: list[str] = Field(max_length=8)

    @field_validator("ids")
    @classmethod
    def validate_ids(cls, values: list[str]) -> list[str]:
        if any(not value or len(value) > 64 for value in values):
            raise ValueError("Некорректный идентификатор проверки.")
        return list(dict.fromkeys(values))


class ReportIllustrationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    sequence: int
    title: str
    caption: str
    alt_text: str
    file_url: str | None


class RunDetail(BaseModel):
    """The intentionally small public view of a run.

    Raw prompts, model identifiers, proxy addresses, headers and response
    bodies remain in the database for diagnostics. They are not part of the
    public or shareable report API.
    """

    model_config = ConfigDict(from_attributes=True)

    id: str
    created_at: datetime
    status: RunStatus
    domain: str | None
    progress_current: int
    progress_total: int
    progress_percent: int
    stage_key: str | None
    stage_label: str | None
    stage_detail: str | None
    eta_seconds: int | None
    run_state: str
    state_revision: int
    state_changed_at: datetime
    checkpointed_at: datetime | None
    queue_position: int | None = None
    queue_total: int | None = None
    attempt_count: int
    resume_count: int
    resume_reason: str | None
    last_resumed_at: datetime | None
    started_at: datetime | None
    finished_at: datetime | None
    analysis_markdown: str | None
    report_json: dict[str, Any] | None
    error_message: str | None
    illustrations: list[ReportIllustrationOut]


def canonical_report_illustrations(
    report_json: dict[str, Any] | None,
) -> list[ReportIllustrationOut]:
    """Read public illustration copy only from the persisted report snapshot."""

    if not isinstance(report_json, dict):
        return []
    raw_illustrations = report_json.get("illustrations")
    if not isinstance(raw_illustrations, list):
        return []

    canonical: list[ReportIllustrationOut] = []
    seen_sequences: set[int] = set()
    for raw_item in raw_illustrations:
        try:
            item = ReportIllustrationOut.model_validate(raw_item)
        except (TypeError, ValidationError):
            continue
        if not item.file_url:
            continue
        if item.sequence in seen_sequences:
            continue
        seen_sequences.add(item.sequence)
        canonical.append(item)
    return sorted(canonical, key=lambda item: item.sequence)


def build_public_run_detail(
    run: Any,
    *,
    queue_position: int | None,
    queue_total: int | None,
) -> RunDetail:
    """Build a detail response without reading the mutable ORM illustration rows."""

    payload = {
        field_name: getattr(run, field_name)
        for field_name in RunDetail.model_fields
        if field_name not in {"illustrations", "queue_position", "queue_total"}
    }
    is_published = has_visible_publication_snapshot(run)
    if not is_published:
        payload["analysis_markdown"] = None
        payload["report_json"] = None
    payload.update(
        {
            "queue_position": queue_position,
            "queue_total": queue_total,
            "illustrations": (
                canonical_report_illustrations(run.report_json) if is_published else []
            ),
        }
    )
    return RunDetail.model_validate(payload)


class ShareTokenResponse(BaseModel):
    share_token: str
    share_url: str


class RetryRunResponse(BaseModel):
    ok: bool
    run_id: str
