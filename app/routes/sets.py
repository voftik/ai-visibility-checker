from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/sets", tags=["sets"])

SETS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "sets"


def _count_domains(payload: dict[str, Any]) -> int:
    if "domains" in payload and isinstance(payload["domains"], list):
        if payload["domains"] and isinstance(payload["domains"][0], dict):
            return len(payload["domains"])
        return len(payload["domains"])
    if "categories" in payload and isinstance(payload["categories"], list):
        n = 0
        for cat in payload["categories"]:
            n += len(cat.get("domains") or [])
        return n
    return 0


def _read_set(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


@router.get("")
async def list_sets() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not SETS_DIR.exists():
        return out
    for p in sorted(SETS_DIR.glob("*.json")):
        data = _read_set(p)
        if not data:
            continue
        out.append({
            "id": data.get("id") or p.stem,
            "title": data.get("title") or p.stem,
            "description": data.get("description") or "",
            "domain_count": _count_domains(data),
            "kind": "categorical" if "categories" in data else "flat",
        })
    return out


@router.get("/{set_id}")
async def get_set(set_id: str) -> dict[str, Any]:
    if "/" in set_id or ".." in set_id:
        raise HTTPException(status_code=404, detail="set not found")
    path = SETS_DIR / f"{set_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="set not found")
    data = _read_set(path)
    if data is None:
        raise HTTPException(status_code=500, detail="set unreadable")
    return data
