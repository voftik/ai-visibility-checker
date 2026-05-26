from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.models import ProbeType, RunStatus


class RunConfig(BaseModel):
    domains: list[str] = Field(default_factory=list)
    user_agents: list[str] = Field(default_factory=list, description="UA labels to use")
    concurrency: int = 8
    timeout_seconds: int = 15


class CreateRunRequest(BaseModel):
    domains: list[str] = Field(default_factory=list)
    user_agents: list[str] = Field(default_factory=list)
    concurrency: int | None = None
    timeout_seconds: int | None = None
    source_breakdown: dict[str, Any] | None = None


class CreateRunResponse(BaseModel):
    run_id: str


class RunSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    created_at: datetime
    status: RunStatus
    progress_current: int
    progress_total: int
    share_token: str | None = None


class DomainProbeOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    domain: str
    user_agent_label: str
    user_agent_string: str
    target_url: str
    probe_type: ProbeType
    http_status: int | None
    response_size_bytes: int | None
    ttfb_ms: int | None
    total_time_ms: int | None
    tls_ok: bool | None
    final_url: str | None
    redirect_chain: list[Any] | None
    response_headers: dict[str, Any] | None
    detected_protections: list[Any] | None
    challenge_detected: bool
    body_sample: str | None
    body_looks_empty: bool
    content_extractable_text_length: int | None = None
    content_signals: dict[str, Any] | None = None
    error_class: str | None
    error_message: str | None
    created_at: datetime


class RobotsRuleOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    domain: str
    bot_name: str
    rule: str
    raw_directives: str | None


class RunDetail(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    created_at: datetime
    status: RunStatus
    config_json: dict[str, Any]
    progress_current: int
    progress_total: int
    analysis_markdown: str | None
    error_message: str | None
    share_token: str | None = None
    probes: list[DomainProbeOut]
    robots_rules: list[RobotsRuleOut]


class ShareTokenResponse(BaseModel):
    share_token: str
    share_url: str


class ShareRevokeResponse(BaseModel):
    ok: bool
