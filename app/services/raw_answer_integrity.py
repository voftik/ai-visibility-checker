"""Canonical integrity fingerprints for immutable model-panel answers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

RAW_ANSWER_FINGERPRINT_VERSION = "aiv-raw-answer-fingerprint-v1"


def model_answer_fingerprint_rows(rows: Sequence[Any]) -> str:
    """Hash every persisted raw-answer field in a stable physical-row order.

    The function deliberately includes database identity and ``created_at`` in
    addition to the semantic response fields.  Saved-answer reprocessing uses
    this digest as an exact no-write boundary, not merely as a content checksum.
    """

    def value(row: Any, key: str) -> object:
        return row.get(key) if isinstance(row, Mapping) else getattr(row, key)

    payload = [
        {
            "id": value(row, "id"),
            "run_id": value(row, "run_id"),
            "prompt_id": value(row, "prompt_id"),
            "provider_key": value(row, "provider_key"),
            "model": value(row, "model"),
            "mode": value(row, "mode"),
            "status": value(row, "status"),
            "response_text": value(row, "response_text"),
            "citations_json": value(row, "citations_json"),
            "usage_json": value(row, "usage_json"),
            "error_message": value(row, "error_message"),
            "created_at": (
                value(row, "created_at").isoformat()
                if value(row, "created_at") is not None
                else None
            ),
        }
        for row in sorted(rows, key=lambda item: int(value(item, "id")))
    ]
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
