from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.db import init_db
from app.routes.pages import router as pages_router
from app.routes.runs import router as runs_router
from app.routes.shared import router as shared_router
from app.services.event_bus import bus
from app.services import proxy_pool
from app.services.run_coordinator import coordinator

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
PROXY_CACHE_PATH = Path(__file__).resolve().parent.parent / ".proxy_cache.json"


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    bus.start_cleanup()

    # Outbound proxy pool. Empty WEBSHARE_API_KEY (or PROXY_ENABLED=false)
    # keeps the service running in direct mode — identical to pre-proxy
    # behaviour, which is exactly what we want when the feature is opt-out.
    if settings.PROXY_ENABLED and settings.WEBSHARE_API_KEY:
        pool = proxy_pool.ProxyPool(
            api_key=settings.WEBSHARE_API_KEY,
            cache_path=PROXY_CACHE_PATH,
            cooldown_seconds=settings.PROXY_COOLDOWN_SECONDS,
            refresh_interval_seconds=settings.PROXY_REFRESH_INTERVAL_SECONDS,
        )
        loaded = await pool.load()
        logger.info("ProxyPool: loaded %d proxies on startup", loaded)
        pool.start_background_refresh()
        proxy_pool.set_pool(pool)
    else:
        logger.info("ProxyPool: disabled (WEBSHARE_API_KEY empty or PROXY_ENABLED=false)")
        proxy_pool.set_pool(None)

    await coordinator.start()
    try:
        yield
    finally:
        await coordinator.stop()
        pool = proxy_pool.get_pool()
        if pool is not None:
            pool.stop_background_refresh()


app = FastAPI(
    title="AIV",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

app.include_router(pages_router)
app.include_router(runs_router)
app.include_router(shared_router)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/api/healthz", include_in_schema=False)
async def healthz() -> dict:
    """Public liveness check without infrastructure details."""
    return {"ok": True}
