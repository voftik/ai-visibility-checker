from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.db import init_db
from app.routes.defaults import router as defaults_router
from app.routes.pages import router as pages_router
from app.routes.runs import router as runs_router
from app.routes.sets import router as sets_router
from app.services.event_bus import bus

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    bus.start_cleanup()
    yield


app = FastAPI(title="AI Visibility Checker", lifespan=lifespan)

app.include_router(pages_router)
app.include_router(runs_router)
app.include_router(defaults_router)
app.include_router(sets_router)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
