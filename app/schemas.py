from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models import RunStatus


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


class ShareTokenResponse(BaseModel):
    share_token: str
    share_url: str


class RetryRunResponse(BaseModel):
    ok: bool
    run_id: str
