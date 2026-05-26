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
def _enable_sqlite_foreign_keys(dbapi_connection, connection_record) -> None:
    del connection_record
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA foreign_keys=ON")
    finally:
        cursor.close()


_ALTER_STATEMENTS: tuple[str, ...] = (
    "ALTER TABLE domain_probes ADD COLUMN content_extractable_text_length INTEGER",
    "ALTER TABLE domain_probes ADD COLUMN content_signals JSON",
    # SQLite refuses inline UNIQUE on ALTER TABLE ADD COLUMN. Add the column
    # plain, then enforce uniqueness through the UNIQUE INDEX below.
    "ALTER TABLE runs ADD COLUMN share_token VARCHAR(64)",
)


_INDEX_STATEMENTS: tuple[str, ...] = (
    "CREATE UNIQUE INDEX IF NOT EXISTS ix_runs_share_token ON runs(share_token)",
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
        for stmt in _INDEX_STATEMENTS:
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
        except Exception:
            pass
        # Recover from previous abrupt shutdowns. Any run still in pending /
        # crawling / analyzing belongs to an asyncio task that died with the
        # process. Mark it failed so the UI doesn't show a forever-spinning
        # entry that will never reach completion.
        try:
            await conn.exec_driver_sql(
                "UPDATE runs SET status = 'failed', "
                "error_message = COALESCE(error_message, "
                "'Прогон прерван перезапуском сервиса. Запустите проверку заново.') "
                "WHERE status IN ('pending', 'crawling', 'analyzing')"
            )
        except Exception:
            pass


async def get_session() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session
