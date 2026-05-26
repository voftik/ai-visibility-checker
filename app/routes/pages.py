from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter()

INDEX_HTML = Path(__file__).resolve().parent.parent.parent / "static" / "index.html"


@router.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(INDEX_HTML, media_type="text/html")


@router.get("/r/{token}", include_in_schema=False)
async def shared_page(token: str) -> FileResponse:
    # Same SPA shell — Alpine init() reads location.pathname and switches into
    # the shared_report tab, then fetches /api/shared/{token}.
    del token  # routing-only; the SPA reads the token from window.location.
    return FileResponse(INDEX_HTML, media_type="text/html")
