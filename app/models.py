from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum as SAEnum,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class RunStatus(str, enum.Enum):
    pending = "pending"
    crawling = "crawling"
    analyzing = "analyzing"
    completed = "completed"
    failed = "failed"


class ProbeType(str, enum.Enum):
    main_page = "main_page"
    robots_txt = "robots_txt"


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    status: Mapped[RunStatus] = mapped_column(
        SAEnum(RunStatus, name="run_status"), default=RunStatus.pending, nullable=False
    )
    config_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    domain: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    progress_current: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    progress_total: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    progress_percent: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    stage_key: Mapped[str | None] = mapped_column(String(64), nullable=True)
    stage_label: Mapped[str | None] = mapped_column(String(160), nullable=True)
    stage_detail: Mapped[str | None] = mapped_column(String(500), nullable=True)
    eta_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    execution_slot: Mapped[int | None] = mapped_column(Integer, nullable=True)
    lease_owner: Mapped[str | None] = mapped_column(String(96), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    heartbeat_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    state_revision: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    state_changed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        nullable=False,
    )
    checkpointed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    attempt_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    resume_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    resume_reason: Mapped[str | None] = mapped_column(
        String(160),
        nullable=True,
    )
    last_resumed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    analysis_markdown: Mapped[str | None] = mapped_column(Text, nullable=True)
    report_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    share_token: Mapped[str | None] = mapped_column(
        String(64), unique=True, nullable=True, index=True
    )

    probes: Mapped[list["DomainProbe"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", passive_deletes=True
    )
    robots_rules: Mapped[list["RobotsRule"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", passive_deletes=True
    )
    pages: Mapped[list["SitePage"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", passive_deletes=True
    )
    artifacts: Mapped[list["RunArtifact"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", passive_deletes=True
    )
    recovery_epochs: Mapped[list["RecoveryEpoch"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", passive_deletes=True
    )
    visibility_prompts: Mapped[list["VisibilityPrompt"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", passive_deletes=True
    )
    model_answers: Mapped[list["ModelAnswer"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", passive_deletes=True
    )
    illustrations: Mapped[list["ReportIllustration"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", passive_deletes=True
    )

    @property
    def run_state(self) -> str:
        if self.status == RunStatus.completed:
            return "completed"
        if self.status == RunStatus.failed:
            return "failed"
        if self.stage_key == "recovering":
            return "recovering"
        if self.status in {RunStatus.crawling, RunStatus.analyzing}:
            return "running"
        if self.execution_slot is not None:
            return "running"
        return "queued"


class DomainProbe(Base):
    __tablename__ = "domain_probes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    domain: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    user_agent_label: Mapped[str] = mapped_column(String(64), nullable=False)
    user_agent_string: Mapped[str] = mapped_column(String(512), nullable=False)
    target_url: Mapped[str] = mapped_column(String(2048), nullable=False)
    page_kind: Mapped[str | None] = mapped_column(String(64), nullable=True)
    probe_type: Mapped[ProbeType] = mapped_column(SAEnum(ProbeType, name="probe_type"), nullable=False)

    http_status: Mapped[int | None] = mapped_column(Integer, nullable=True)
    response_size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ttfb_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tls_ok: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    final_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    redirect_chain: Mapped[list | None] = mapped_column(JSON, nullable=True)
    response_headers: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    detected_protections: Mapped[list | None] = mapped_column(JSON, nullable=True)
    challenge_detected: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    body_sample: Mapped[str | None] = mapped_column(Text, nullable=True)
    body_looks_empty: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    content_extractable_text_length: Mapped[int | None] = mapped_column(Integer, nullable=True)
    content_signals: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error_class: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)

    run: Mapped[Run] = relationship(back_populates="probes")


class RobotsRule(Base):
    __tablename__ = "robots_rules"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    domain: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    bot_name: Mapped[str] = mapped_column(String(64), nullable=False)
    rule: Mapped[str] = mapped_column(String(32), nullable=False)
    raw_directives: Mapped[str | None] = mapped_column(Text, nullable=True)

    run: Mapped[Run] = relationship(back_populates="robots_rules")


class SitePage(Base):
    __tablename__ = "site_pages"
    __table_args__ = (UniqueConstraint("run_id", "url", name="uq_site_pages_run_url"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    url: Mapped[str] = mapped_column(String(2048), nullable=False)
    canonical_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    page_kind: Mapped[str] = mapped_column(String(64), default="other", nullable=False)
    http_status: Mapped[int | None] = mapped_column(Integer, nullable=True)
    title: Mapped[str | None] = mapped_column(String(500), nullable=True)
    meta_description: Mapped[str | None] = mapped_column(Text, nullable=True)
    main_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    text_length: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    content_signals: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)

    run: Mapped[Run] = relationship(back_populates="pages")


class RunArtifact(Base):
    __tablename__ = "run_artifacts"
    __table_args__ = (
        UniqueConstraint("run_id", "artifact_key", name="uq_run_artifacts_run_key"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    stage_key: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    artifact_key: Mapped[str] = mapped_column(String(160), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)
    model: Mapped[str | None] = mapped_column(String(160), nullable=True)
    prompt_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    input_json: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
    output_json: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
    raw_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    usage_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )

    run: Mapped[Run] = relationship(back_populates="artifacts")


class RecoveryEpoch(Base):
    """Append-only decision record for one bounded recovery attempt."""

    __tablename__ = "recovery_epochs"
    __table_args__ = (
        UniqueConstraint("run_id", "epoch", name="uq_recovery_epochs_run_epoch"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    epoch: Mapped[int] = mapped_column(Integer, nullable=False)
    stage_key: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    failure_class: Mapped[str] = mapped_column(String(64), nullable=False)
    failure_code: Mapped[str] = mapped_column(String(96), nullable=False)
    failure_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    facts_digest: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="diagnosing")
    model: Mapped[str | None] = mapped_column(String(160), nullable=True)
    input_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    plan_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    plan_digest: Mapped[str | None] = mapped_column(String(64), nullable=True)
    usage_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    outcome_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )

    run: Mapped[Run] = relationship(back_populates="recovery_epochs")


class VisibilityPrompt(Base):
    __tablename__ = "visibility_prompts"
    __table_args__ = (
        UniqueConstraint("run_id", "prompt_key", name="uq_visibility_prompts_run_key"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    prompt_key: Mapped[str] = mapped_column(String(64), nullable=False)
    intent_class: Mapped[str] = mapped_column(String(8), nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)

    run: Mapped[Run] = relationship(back_populates="visibility_prompts")
    answers: Mapped[list["ModelAnswer"]] = relationship(
        back_populates="prompt", cascade="all, delete-orphan", passive_deletes=True
    )


class ModelAnswer(Base):
    __tablename__ = "model_answers"
    __table_args__ = (
        UniqueConstraint(
            "run_id",
            "prompt_id",
            "provider_key",
            "mode",
            name="uq_model_answers_run_prompt_provider_mode",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    prompt_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("visibility_prompts.id", ondelete="CASCADE"), nullable=False, index=True
    )
    provider_key: Mapped[str] = mapped_column(String(32), nullable=False)
    model: Mapped[str] = mapped_column(String(160), nullable=False)
    mode: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)
    response_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    citations_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    usage_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)

    run: Mapped[Run] = relationship(back_populates="model_answers")
    prompt: Mapped[VisibilityPrompt] = relationship(back_populates="answers")
    annotation: Mapped["AnswerAnnotation | None"] = relationship(
        back_populates="answer",
        cascade="all, delete-orphan",
        passive_deletes=True,
        uselist=False,
    )


class AnswerAnnotation(Base):
    __tablename__ = "answer_annotations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    answer_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("model_answers.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    annotation_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)

    answer: Mapped[ModelAnswer] = relationship(back_populates="annotation")


class ReportIllustration(Base):
    __tablename__ = "report_illustrations"
    __table_args__ = (
        UniqueConstraint("run_id", "sequence", name="uq_report_illustrations_run_sequence"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    caption: Mapped[str] = mapped_column(Text, nullable=False)
    alt_text: Mapped[str] = mapped_column(Text, nullable=False)
    file_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    generation_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str] = mapped_column(String(160), nullable=False)
    usage_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)

    run: Mapped[Run] = relationship(back_populates="illustrations")
