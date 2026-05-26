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
    progress_current: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    progress_total: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    analysis_markdown: Mapped[str | None] = mapped_column(Text, nullable=True)
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
