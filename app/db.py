from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models import Base

DB_PATH = Path(__file__).resolve().parent.parent / "sqlite.db"
DATABASE_URL = f"sqlite+aiosqlite:///{DB_PATH}"

engine = create_async_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


@event.listens_for(engine.sync_engine, "connect")
def _configure_sqlite_connection(dbapi_connection, connection_record) -> None:
    del connection_record
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=10000")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
    finally:
        cursor.close()


_ALTER_STATEMENTS: tuple[str, ...] = (
    "ALTER TABLE domain_probes ADD COLUMN content_extractable_text_length INTEGER",
    "ALTER TABLE domain_probes ADD COLUMN content_signals JSON",
    "ALTER TABLE domain_probes ADD COLUMN page_kind VARCHAR(64)",
    # SQLite refuses inline UNIQUE on ALTER TABLE ADD COLUMN. Add the column
    # plain, then enforce uniqueness through the UNIQUE INDEX below.
    "ALTER TABLE runs ADD COLUMN share_token VARCHAR(64)",
    "ALTER TABLE runs ADD COLUMN domain VARCHAR(255)",
    "ALTER TABLE runs ADD COLUMN progress_percent INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE runs ADD COLUMN stage_key VARCHAR(64)",
    "ALTER TABLE runs ADD COLUMN stage_label VARCHAR(160)",
    "ALTER TABLE runs ADD COLUMN stage_detail VARCHAR(500)",
    "ALTER TABLE runs ADD COLUMN eta_seconds INTEGER",
    "ALTER TABLE runs ADD COLUMN execution_slot INTEGER",
    "ALTER TABLE runs ADD COLUMN lease_owner VARCHAR(96)",
    "ALTER TABLE runs ADD COLUMN lease_expires_at DATETIME",
    "ALTER TABLE runs ADD COLUMN heartbeat_at DATETIME",
    "ALTER TABLE runs ADD COLUMN state_revision INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE runs ADD COLUMN state_changed_at DATETIME",
    "ALTER TABLE runs ADD COLUMN checkpointed_at DATETIME",
    "ALTER TABLE runs ADD COLUMN attempt_count INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE runs ADD COLUMN resume_count INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE runs ADD COLUMN resume_reason VARCHAR(160)",
    "ALTER TABLE runs ADD COLUMN last_resumed_at DATETIME",
    "ALTER TABLE runs ADD COLUMN started_at DATETIME",
    "ALTER TABLE runs ADD COLUMN finished_at DATETIME",
    "ALTER TABLE runs ADD COLUMN report_json JSON",
    "ALTER TABLE report_illustrations ADD COLUMN caption TEXT NOT NULL DEFAULT ''",
)


_INDEX_STATEMENTS: tuple[str, ...] = (
    "CREATE UNIQUE INDEX IF NOT EXISTS ix_runs_share_token ON runs(share_token)",
    "CREATE INDEX IF NOT EXISTS ix_runs_domain ON runs(domain)",
    (
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_runs_execution_slot "
        "ON runs(execution_slot) WHERE execution_slot IS NOT NULL"
    ),
)


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Idempotent column-add for pre-existing DBs. SQLite raises if the
        # column already exists; we swallow that and continue.
        for stmt in _ALTER_STATEMENTS:
            try:
                await conn.exec_driver_sql(stmt)
            except Exception:
                pass
        # Clean up orphan rows left by older versions that relied on SQLite
        # cascades while PRAGMA foreign_keys was still disabled.
        try:
            await conn.exec_driver_sql(
                "DELETE FROM domain_probes "
                "WHERE run_id NOT IN (SELECT id FROM runs)"
            )
            await conn.exec_driver_sql(
                "DELETE FROM robots_rules "
                "WHERE run_id NOT IN (SELECT id FROM runs)"
            )
            await conn.exec_driver_sql(
                "UPDATE runs SET domain = json_extract(config_json, '$.domains[0]') "
                "WHERE domain IS NULL"
            )
            await conn.exec_driver_sql(
                "UPDATE runs SET state_changed_at = COALESCE("
                "state_changed_at, created_at, CURRENT_TIMESTAMP)"
            )
        except Exception:
            pass
        # Heal impossible slot states before installing the cross-process
        # uniqueness guard. This is only relevant when upgrading a database
        # that briefly ran an older queue implementation without the index.
        try:
            await conn.exec_driver_sql(
                "UPDATE runs SET execution_slot = NULL, lease_owner = NULL, "
                "lease_expires_at = NULL, heartbeat_at = NULL "
                "WHERE status IN ('completed', 'failed')"
            )
            await conn.exec_driver_sql(
                "UPDATE runs SET status = 'pending', execution_slot = NULL, "
                "lease_owner = NULL, lease_expires_at = NULL, "
                "heartbeat_at = NULL, stage_key = 'recovering', "
                "stage_label = 'Восстанавливаем проверку', "
                "stage_detail = 'Продолжим с уже сохранённых данных.', "
                "eta_seconds = NULL, "
                "resume_count = COALESCE(resume_count, 0) + 1, "
                "resume_reason = 'invalid_slot_repair', "
                "last_resumed_at = CURRENT_TIMESTAMP, "
                "state_revision = COALESCE(state_revision, 0) + 1, "
                "state_changed_at = CURRENT_TIMESTAMP "
                "WHERE execution_slot IS NOT NULL AND execution_slot != 1"
            )
            await conn.exec_driver_sql(
                "UPDATE runs SET status = 'pending', execution_slot = NULL, "
                "lease_owner = NULL, lease_expires_at = NULL, "
                "heartbeat_at = NULL, stage_key = 'recovering', "
                "stage_label = 'Восстанавливаем проверку', "
                "stage_detail = 'Продолжим с уже сохранённых данных.', "
                "eta_seconds = NULL, "
                "resume_count = COALESCE(resume_count, 0) + 1, "
                "resume_reason = 'slot_repair', "
                "last_resumed_at = CURRENT_TIMESTAMP, "
                "state_revision = COALESCE(state_revision, 0) + 1, "
                "state_changed_at = CURRENT_TIMESTAMP "
                "WHERE execution_slot IS NOT NULL AND id NOT IN ("
                "SELECT id FROM runs WHERE execution_slot IS NOT NULL "
                "ORDER BY COALESCE(heartbeat_at, created_at) DESC, id DESC "
                "LIMIT 1)"
            )
        except Exception:
            pass
        for stmt in _INDEX_STATEMENTS:
            try:
                await conn.exec_driver_sql(stmt)
            except Exception:
                # Existing optional indexes are best-effort; the execution
                # slot index is validated explicitly below.
                if "uq_runs_execution_slot" in stmt:
                    raise
        index_rows = (
            await conn.exec_driver_sql("PRAGMA index_list('runs')")
        ).all()
        slot_indexes = {
            str(row[1]): bool(row[2])
            for row in index_rows
        }
        if slot_indexes.get("uq_runs_execution_slot") is not True:
            raise RuntimeError(
                "Durable run queue requires unique execution-slot index"
            )
        # Old releases had no durable lease. Requeue only those legacy active
        # rows here; the coordinator below handles current expired leases.
        try:
            await conn.exec_driver_sql(
                "UPDATE runs SET status = 'pending', "
                "execution_slot = NULL, lease_owner = NULL, "
                "lease_expires_at = NULL, heartbeat_at = NULL, "
                "stage_key = 'recovering', "
                "stage_label = 'Восстанавливаем проверку', "
                "stage_detail = 'Продолжим с уже сохранённых данных.', "
                "eta_seconds = NULL, error_message = NULL, "
                "resume_count = COALESCE(resume_count, 0) + 1, "
                "resume_reason = 'service_restart', "
                "last_resumed_at = CURRENT_TIMESTAMP, "
                "state_revision = COALESCE(state_revision, 0) + 1, "
                "state_changed_at = CURRENT_TIMESTAMP "
                "WHERE status IN ('crawling', 'analyzing') "
                "AND (execution_slot IS NULL OR lease_owner IS NULL "
                "OR lease_expires_at IS NULL)"
            )
            await conn.exec_driver_sql(
                "UPDATE runs SET execution_slot = NULL, lease_owner = NULL, "
                "lease_expires_at = NULL, heartbeat_at = NULL "
                "WHERE status IN ('completed', 'failed')"
            )
        except Exception:
            pass


async def get_session() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session
