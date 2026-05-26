from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

router = APIRouter(prefix="/api/defaults", tags=["defaults"])

DOMAINS_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "domains_default.txt"


@router.get("/domains")
async def get_default_domains() -> dict[str, list[str]]:
    if not DOMAINS_FILE.exists():
        return {"domains": []}
    raw = DOMAINS_FILE.read_text(encoding="utf-8")
    out: list[str] = []
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return {"domains": out}
