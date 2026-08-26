from __future__ import annotations

import asyncio
import sqlite3
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

from app.services.sqlite_lease_heartbeat import SQLiteLeaseHeartbeat


class SQLiteLeaseHeartbeatTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._temp_dir.name) / "lease.sqlite3"
        connection = sqlite3.connect(self.db_path)
        connection.execute(
            "CREATE TABLE runs ("
            "id TEXT PRIMARY KEY, status TEXT NOT NULL, "
            "execution_slot INTEGER, lease_owner TEXT, "
            "heartbeat_at TEXT, lease_expires_at TEXT)"
        )
        connection.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?)",
            ("run-1", "analyzing", 1, "owner-1", None, None),
        )
        connection.commit()
        connection.close()

    def tearDown(self) -> None:
        self._temp_dir.cleanup()

    def _row(self) -> tuple[object, ...]:
        connection = sqlite3.connect(self.db_path)
        try:
            return connection.execute(
                "SELECT status, execution_slot, lease_owner, "
                "heartbeat_at, lease_expires_at FROM runs WHERE id='run-1'"
            ).fetchone()
        finally:
            connection.close()

    async def test_renews_while_the_event_loop_is_cpu_blocked(self) -> None:
        heartbeat = SQLiteLeaseHeartbeat(
            database_path=self.db_path,
            run_id="run-1",
            owner="owner-1",
            lease_seconds=0.3,
            interval_seconds=0.05,
        )
        heartbeat.start()
        try:
            time.sleep(0.55)  # noqa: ASYNC251 - deliberately starve event loop
            snapshot = heartbeat.snapshot()
            self.assertGreaterEqual(snapshot.renewals, 4)
            row = self._row()
            self.assertEqual(row[:3], ("analyzing", 1, "owner-1"))
            lease_expires_at = datetime.fromisoformat(str(row[4])).replace(
                tzinfo=timezone.utc
            )
            self.assertGreater(lease_expires_at, datetime.now(timezone.utc))

            connection = sqlite3.connect(self.db_path)
            try:
                recovered = connection.execute(
                    "UPDATE runs SET status='pending', execution_slot=NULL, "
                    "lease_owner=NULL WHERE id='run-1' AND lease_expires_at<=?",
                    (
                        datetime.now(timezone.utc)
                        .replace(tzinfo=None)
                        .isoformat(sep=" ", timespec="microseconds"),
                    ),
                )
                connection.commit()
            finally:
                connection.close()
            self.assertEqual(recovered.rowcount, 0)
        finally:
            heartbeat.stop()
        self.assertFalse(heartbeat.running)

    async def test_signals_terminal_or_foreign_ownership_without_overwrite(
        self,
    ) -> None:
        heartbeat = SQLiteLeaseHeartbeat(
            database_path=self.db_path,
            run_id="run-1",
            owner="wrong-owner",
            lease_seconds=0.3,
            interval_seconds=0.05,
        )
        heartbeat.start()
        try:
            await asyncio.wait_for(heartbeat.wait_lost(), timeout=1)
            self.assertTrue(heartbeat.lost)
            self.assertEqual(self._row()[:3], ("analyzing", 1, "owner-1"))
        finally:
            heartbeat.stop()
        self.assertFalse(heartbeat.running)

    async def test_permanent_operational_error_fails_closed(self) -> None:
        broken_path = Path(self._temp_dir.name) / "missing-runs.sqlite3"
        sqlite3.connect(broken_path).close()
        heartbeat = SQLiteLeaseHeartbeat(
            database_path=broken_path,
            run_id="run-1",
            owner="owner-1",
            lease_seconds=0.3,
            interval_seconds=0.05,
        )
        heartbeat.start()
        try:
            await asyncio.wait_for(heartbeat.wait_lost(), timeout=1)
            snapshot = heartbeat.snapshot()
            self.assertTrue(snapshot.lost)
            self.assertEqual(snapshot.renewals, 0)
            self.assertIn("no such table", snapshot.last_error or "")
        finally:
            heartbeat.stop()
        self.assertFalse(heartbeat.running)

    async def test_contention_fails_closed_before_lease_can_expire(self) -> None:
        blocker = sqlite3.connect(self.db_path, isolation_level=None)
        blocker.execute("BEGIN EXCLUSIVE")
        heartbeat = SQLiteLeaseHeartbeat(
            database_path=self.db_path,
            run_id="run-1",
            owner="owner-1",
            lease_seconds=0.3,
            interval_seconds=0.03,
            busy_timeout_seconds=1,
        )
        started_at = time.monotonic()
        heartbeat.start()
        try:
            await asyncio.wait_for(heartbeat.wait_lost(), timeout=1)
            elapsed = time.monotonic() - started_at
            snapshot = heartbeat.snapshot()
            self.assertTrue(snapshot.lost)
            self.assertEqual(snapshot.renewals, 0)
            self.assertLess(elapsed, 0.3)
            self.assertIn("locked", (snapshot.last_error or "").casefold())
        finally:
            blocker.rollback()
            blocker.close()
            heartbeat.stop()
        self.assertFalse(heartbeat.running)


if __name__ == "__main__":
    unittest.main()
