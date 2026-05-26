"""In-memory pub/sub for SSE streaming of run logs and progress.

Event shapes (by convention — services should emit one of these):
    {"type": "log",          "level": "info|warn|error", "message": "...", "ts": "..."}
    {"type": "progress",     "current": N, "total": M, "phase": "crawling|analyzing"}
    {"type": "probe_done",   "domain": "...", "user_agent_label": "...",
                             "http_status": ..., "summary": "..."}
    {"type": "phase_change", "phase": "crawling_started|crawling_done|"
                                      "analyzing_started|analyzing_done|completed|failed"}
    {"type": "final",        "status": "completed|failed"}

Each channel keeps a ring-buffer of the last `_HISTORY_MAX` events so that an
SSE subscriber that reconnects (browser moved between visibility states, nginx
idle drop, transient network) can replay missed events using the standard
`Last-Event-ID` header. `subscribe()` yields `(seq_id, event)` pairs; the SSE
route writes `id: <seq_id>` on the wire so the browser's native EventSource
includes that ID on every reconnect attempt.
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# How many recent events to keep per channel for replay on reconnect.
# 500 covers ~5 minutes of probe-heavy crawling on a 50-domain × 6-UA run.
_HISTORY_MAX = 500


@dataclass
class _Channel:
    queues: list[asyncio.Queue[tuple[int, dict[str, Any]] | None]] = field(default_factory=list)
    history: deque[tuple[int, dict[str, Any]]] = field(default_factory=lambda: deque(maxlen=_HISTORY_MAX))
    seq: int = 0
    finished: bool = False
    finished_at: float | None = None


class EventBus:
    def __init__(self) -> None:
        self._channels: dict[str, _Channel] = {}
        self._cleanup_task: asyncio.Task[None] | None = None

    def _channel(self, run_id: str) -> _Channel:
        ch = self._channels.get(run_id)
        if ch is None:
            ch = _Channel()
            self._channels[run_id] = ch
        return ch

    def channel_history(self, run_id: str) -> list[tuple[int, dict[str, Any]]]:
        """Return a snapshot of the channel's history (used for one-shot replay)."""
        ch = self._channels.get(run_id)
        if ch is None:
            return []
        return list(ch.history)

    def publish(self, run_id: str, event: dict[str, Any]) -> None:
        if "ts" not in event:
            event = {**event, "ts": datetime.now(timezone.utc).isoformat()}
        ch = self._channel(run_id)
        ch.seq += 1
        item = (ch.seq, event)
        ch.history.append(item)
        for q in list(ch.queues):
            q.put_nowait(item)
        if event.get("type") == "final":
            ch.finished = True
            ch.finished_at = time.monotonic()
            for q in list(ch.queues):
                q.put_nowait(None)

    async def subscribe(
        self,
        run_id: str,
        last_event_id: int | None = None,
    ) -> AsyncIterator[tuple[int, dict[str, Any]]]:
        """Yield (seq_id, event) pairs.

        If `last_event_id` is provided, missed events with seq > last_event_id are
        replayed first from history, then the live stream is attached.
        If `last_event_id` is None, the full current history is replayed first
        (a brand-new tab catches up on whatever it missed before subscribing).
        """
        ch = self._channel(run_id)
        q: asyncio.Queue[tuple[int, dict[str, Any]] | None] = asyncio.Queue()
        # CRITICAL: snapshot + queue-attach must happen without any awaits in
        # between, so publish() (which is sync) cannot interleave and drop an
        # event between the two operations. asyncio is single-threaded → this
        # whole block is atomic.
        snapshot = list(ch.history)
        ch.queues.append(q)
        replayed_seqs: set[int] = set()

        try:
            for seq, event in snapshot:
                if last_event_id is None or seq > last_event_id:
                    yield (seq, event)
                    replayed_seqs.add(seq)

            # If the channel was already finished and history covers the final
            # event, bail out without waiting for live events.
            if ch.finished and snapshot and snapshot[-1][1].get("type") == "final":
                return

            while True:
                item = await q.get()
                if item is None:
                    return
                seq, event = item
                if seq in replayed_seqs:
                    # The same event arrived through the live queue right after
                    # we replayed it from history. Skip the duplicate.
                    continue
                yield (seq, event)
        finally:
            if q in ch.queues:
                ch.queues.remove(q)

    def mark_finished_if_absent(self, run_id: str) -> None:
        """Used when a run finished before any subscribers; ensures cleanup timer kicks in."""
        ch = self._channel(run_id)
        if not ch.finished:
            ch.finished = True
            ch.finished_at = time.monotonic()

    async def _cleanup_loop(self) -> None:
        # Every 5 minutes, drop channels that finished more than 1 hour ago.
        while True:
            await asyncio.sleep(300)
            now = time.monotonic()
            for run_id in list(self._channels.keys()):
                ch = self._channels[run_id]
                if ch.finished and ch.finished_at and now - ch.finished_at > 3600 and not ch.queues:
                    del self._channels[run_id]

    def start_cleanup(self) -> None:
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())


bus = EventBus()
