from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse

router = APIRouter()

INDEX_HTML = Path(__file__).resolve().parent.parent.parent / "static" / "index.html"
UI_BUILD_ID = "2026-07-31.30"
NO_STORE_HEADERS = {
    "Cache-Control": "no-store, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
    "X-AIV-UI-Version": UI_BUILD_ID,
}


@router.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(
        INDEX_HTML,
        media_type="text/html",
        headers=NO_STORE_HEADERS,
    )


@router.get("/history", include_in_schema=False)
async def history_page() -> FileResponse:
    return FileResponse(
        INDEX_HTML,
        media_type="text/html",
        headers=NO_STORE_HEADERS,
    )


@router.get("/r/{token}", include_in_schema=False)
async def shared_page(token: str) -> FileResponse:
    # Same SPA shell — Alpine init() reads location.pathname and switches into
    # the shared_report tab, then fetches /api/shared/{token}.
    del token  # routing-only; the SPA reads the token from window.location.
    return FileResponse(
        INDEX_HTML,
        media_type="text/html",
        headers=NO_STORE_HEADERS,
    )


@router.get("/api/ui-version", include_in_schema=False)
async def ui_version() -> JSONResponse:
    """Let an already-open report detect a newer interface build."""
    return JSONResponse(
        {"version": UI_BUILD_ID},
        headers=NO_STORE_HEADERS,
    )
