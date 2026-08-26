"""Lease renewal that remains alive while an asyncio worker is CPU-bound.

The analysis pipeline contains deliberately lossless validation steps.  Some
of them are synchronous and can occupy the worker event loop long enough to
starve an asyncio heartbeat.  A second process would then mistake healthy work
for an abandoned lease.  This module keeps the ownership-fenced SQLite update
in a tiny dedicated thread; it never advances pipeline state and cannot claim
or transfer ownership.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


def database_path_for_engine(engine: Any) -> Path:
    """Return the file backing a SQLAlchemy SQLite engine.

    Tests replace the module-level async engine with a temporary one, so the
    path must be resolved from the live engine instead of importing ``DB_PATH``.
    """

    database = engine.url.database
    if not database or database == ":memory:":
        raise ValueError("A file-backed SQLite database is required for leases")
    return Path(database).expanduser().resolve()


def _sqlite_datetime(value: datetime) -> str:
    """Match SQLAlchemy's UTC-naive SQLite DateTime representation."""

    normalized = value.astimezone(UTC).replace(tzinfo=None)
    return normalized.isoformat(sep=" ", timespec="microseconds")


@dataclass(frozen=True)
class LeaseHeartbeatSnapshot:
    renewals: int
    lost: bool
    last_heartbeat_at: datetime | None
    last_error: str | None


class SQLiteLeaseHeartbeat:
    """Renew one already-owned run lease from a dedicated daemon thread."""

    def __init__(
        self,
        *,
        database_path: Path | str,
        run_id: str,
        owner: str,
        lease_seconds: float,
        interval_seconds: float | None = None,
        busy_timeout_seconds: float = 10.0,
    ) -> None:
        self.database_path = Path(database_path).expanduser().resolve()
        self.run_id = str(run_id)
        self.owner = str(owner)
        self.lease_seconds = max(0.2, float(lease_seconds))
        default_interval = min(30.0, max(0.25, self.lease_seconds / 3.0))
        self.interval_seconds = min(
            max(0.05, float(interval_seconds or default_interval)),
            max(0.05, self.lease_seconds / 2.0),
        )
        # SQLite may otherwise wait inside one UPDATE longer than the lease it
        # is supposed to protect. Keep each lock wait comfortably below the
        # renewal deadline; repeated BUSY/LOCKED failures are still retried.
        self.busy_timeout_seconds = min(
            max(0.01, float(busy_timeout_seconds)),
            max(0.01, self.lease_seconds * 0.1),
        )
        self._renewal_deadline_fraction = 0.8
        self._stop_event = threading.Event()
        self._lost_event = threading.Event()
        self._state_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._async_lost_event: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._renewals = 0
        self._last_heartbeat_at: datetime | None = None
        self._last_error: str | None = None

    @property
    def lost(self) -> bool:
        return self._lost_event.is_set()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def snapshot(self) -> LeaseHeartbeatSnapshot:
        with self._state_lock:
            return LeaseHeartbeatSnapshot(
                renewals=self._renewals,
                lost=self._lost_event.is_set(),
                last_heartbeat_at=self._last_heartbeat_at,
                last_error=self._last_error,
            )

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Lease heartbeat has already been started")
        self._loop = asyncio.get_running_loop()
        self._async_lost_event = asyncio.Event()
        self._thread = threading.Thread(
            target=self._run,
            name=f"aiv-lease-heartbeat-{self.run_id[:12]}",
            daemon=True,
        )
        self._thread.start()

    async def wait_lost(self) -> None:
        event = self._async_lost_event
        if event is None:
            raise RuntimeError("Lease heartbeat has not been started")
        await event.wait()

    def stop(self, *, timeout_seconds: float | None = None) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is None:
            return
        timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else self.busy_timeout_seconds + 2.0
        )
        thread.join(max(0.1, float(timeout)))
        if thread.is_alive():
            raise RuntimeError("Lease heartbeat thread did not stop")

    def _signal_lost(self) -> None:
        if self._lost_event.is_set():
            return
        self._lost_event.set()
        loop = self._loop
        event = self._async_lost_event
        if loop is not None and event is not None and not loop.is_closed():
            loop.call_soon_threadsafe(event.set)

    def _record_success(self, now: datetime) -> None:
        with self._state_lock:
            self._renewals += 1
            self._last_heartbeat_at = now
            self._last_error = None

    def _record_error(self, error: BaseException) -> None:
        with self._state_lock:
            self._last_error = f"{type(error).__name__}: {error}"[:1000]

    @staticmethod
    def _is_retryable_operational_error(error: sqlite3.OperationalError) -> bool:
        """Only SQLite contention is safe to retry without losing ownership."""

        error_code = getattr(error, "sqlite_errorcode", None)
        if isinstance(error_code, int) and (error_code & 0xFF) in {
            sqlite3.SQLITE_BUSY,
            sqlite3.SQLITE_LOCKED,
        }:
            return True
        message = str(error).casefold()
        return "locked" in message or "busy" in message

    @staticmethod
    def _rollback_quietly(connection: sqlite3.Connection) -> None:
        try:
            connection.rollback()
        except sqlite3.Error:
            pass

    def _run(self) -> None:
        connection: sqlite3.Connection | None = None
        renew_by = time.monotonic() + (
            self.lease_seconds * self._renewal_deadline_fraction
        )
        try:
            connection = sqlite3.connect(
                str(self.database_path),
                timeout=self.busy_timeout_seconds,
            )
            connection.execute(
                f"PRAGMA busy_timeout={int(self.busy_timeout_seconds * 1000)}"
            )
            while not self._stop_event.is_set():
                now = datetime.now(UTC)
                lease_until = now + timedelta(seconds=self.lease_seconds)
                try:
                    renewed = connection.execute(
                        "UPDATE runs SET heartbeat_at=?, lease_expires_at=? "
                        "WHERE id=? AND execution_slot=1 AND lease_owner=? "
                        "AND status IN ('pending','crawling','analyzing')",
                        (
                            _sqlite_datetime(now),
                            _sqlite_datetime(lease_until),
                            self.run_id,
                            self.owner,
                        ),
                    )
                    connection.commit()
                except sqlite3.OperationalError as error:
                    self._rollback_quietly(connection)
                    self._record_error(error)
                    if not self._is_retryable_operational_error(error):
                        self._signal_lost()
                        break
                    remaining = renew_by - time.monotonic()
                    if remaining <= 0:
                        self._signal_lost()
                        break
                    if self._stop_event.wait(
                        min(1.0, self.interval_seconds, remaining)
                    ):
                        break
                    continue
                except Exception as error:  # noqa: BLE001 - lease must fail closed
                    self._rollback_quietly(connection)
                    self._record_error(error)
                    self._signal_lost()
                    break
                if renewed.rowcount != 1:
                    self._signal_lost()
                    break
                self._record_success(now)
                renew_by = time.monotonic() + (
                    self.lease_seconds * self._renewal_deadline_fraction
                )
                if self._stop_event.wait(self.interval_seconds):
                    break
        except Exception as error:  # noqa: BLE001 - lease must fail closed
            self._record_error(error)
            self._signal_lost()
        finally:
            if connection is not None:
                connection.close()
