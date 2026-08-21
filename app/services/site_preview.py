"""Persist a real first-screen snapshot for a report run."""
from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import json
import logging
import os
import re
import shlex
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select

from app.config import settings
from app.db import SessionLocal
from app.models import RunArtifact
from app.services.run_lease import assert_run_lease

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"
GENERATED_DIR = STATIC_DIR / "generated"
SITE_PREVIEW_ARTIFACT_KEY = "site_preview"
SITE_PREVIEW_VERSION = "site-preview-v1"
SITE_PREVIEW_WIDTH = 1440
SITE_PREVIEW_HEIGHT = 900
SITE_PREVIEW_TIMEOUT_SECONDS = 40
MAX_WORKER_OUTPUT_BYTES = 9 * 1024 * 1024
MAX_JPEG_BYTES = 6 * 1024 * 1024
_FILE_URL_RX = re.compile(
    r"^/static/generated/"
    r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})/"
    r"site-preview-([0-9a-f]{12})\.jpg$"
)
_capture_slot = asyncio.Semaphore(1)


def _safe_run_directory(run_id: str) -> Path:
    canonical = str(uuid.UUID(run_id))
    run_directory = (GENERATED_DIR / canonical).resolve()
    generated_root = GENERATED_DIR.resolve()
    if not run_directory.is_relative_to(generated_root):
        raise ValueError("unsafe_run_path")
    return run_directory


def _path_for_file_url(file_url: str) -> Path | None:
    match = _FILE_URL_RX.fullmatch(file_url or "")
    if match is None:
        return None
    path = (STATIC_DIR / file_url.removeprefix("/static/")).resolve()
    if not path.is_relative_to(GENERATED_DIR.resolve()):
        return None
    return path


def public_site_preview(value: Any, *, require_file: bool = True) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    file_url = str(value.get("file_url") or "")
    file_path = _path_for_file_url(file_url)
    if file_path is None or (require_file and not file_path.is_file()):
        return None
    width = value.get("width")
    height = value.get("height")
    if not isinstance(width, int) or not 1024 <= width <= 1920:
        return None
    if not isinstance(height, int) or not 720 <= height <= 1440:
        return None
    return {
        "state": "captured",
        "file_url": file_url,
        "source_domain": str(value.get("source_domain") or "")[:255],
        "width": width,
        "height": height,
        "captured_at": str(value.get("captured_at") or "")[:64],
        "sha256": str(value.get("sha256") or "")[:64],
    }


async def _artifact(run_id: str) -> RunArtifact | None:
    async with SessionLocal() as session:
        return (
            await session.execute(
                select(RunArtifact).where(
                    RunArtifact.run_id == run_id,
                    RunArtifact.artifact_key == SITE_PREVIEW_ARTIFACT_KEY,
                )
            )
        ).scalar_one_or_none()


async def _save_artifact(
    run_id: str,
    *,
    status: str,
    input_json: dict[str, Any],
    output_json: dict[str, Any] | None = None,
    error_message: str | None = None,
) -> None:
    await assert_run_lease(run_id)
    async with SessionLocal() as session:
        artifact = (
            await session.execute(
                select(RunArtifact).where(
                    RunArtifact.run_id == run_id,
                    RunArtifact.artifact_key == SITE_PREVIEW_ARTIFACT_KEY,
                )
            )
        ).scalar_one_or_none()
        if artifact is None:
            artifact = RunArtifact(
                run_id=run_id,
                stage_key="site_discovery",
                artifact_key=SITE_PREVIEW_ARTIFACT_KEY,
            )
            session.add(artifact)
        artifact.stage_key = "site_discovery"
        artifact.status = status
        artifact.model = "playwright/chromium"
        artifact.prompt_version = SITE_PREVIEW_VERSION
        artifact.input_json = input_json
        artifact.output_json = output_json
        artifact.error_message = error_message[:1000] if error_message else None
        await session.commit()


async def get_saved_site_preview(
    run_id: str,
    *,
    require_file: bool = True,
) -> dict[str, Any] | None:
    artifact = await _artifact(run_id)
    if (
        artifact is None
        or artifact.status != "completed"
        or artifact.prompt_version != SITE_PREVIEW_VERSION
    ):
        return None
    return public_site_preview(artifact.output_json, require_file=require_file)


async def _run_worker(
    url: str,
    *,
    width: int,
    height: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    command = str(settings.SITE_PREVIEW_WORKER_COMMAND or "").strip()
    if not command:
        from app.services.site_preview_worker import capture_preview

        return await capture_preview(
            url,
            width=width,
            height=height,
            timeout_seconds=timeout_seconds,
        )

    process = await asyncio.create_subprocess_exec(
        *shlex.split(command),
        "--url",
        url,
        "--width",
        str(width),
        "--height",
        str(height),
        "--timeout",
        str(timeout_seconds),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout_seconds + 8,
        )
    except asyncio.CancelledError:
        if process.returncode is None:
            process.kill()
            await process.wait()
        raise
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        raise TimeoutError("site_preview_worker_timeout")
    if len(stdout) > MAX_WORKER_OUTPUT_BYTES:
        raise ValueError("site_preview_worker_output_too_large")
    if process.returncode != 0:
        detail = stderr.decode("utf-8", errors="replace").strip()
        if not detail:
            detail = stdout.decode("utf-8", errors="replace").strip()
        raise RuntimeError(detail[:1000] or "site_preview_worker_failed")
    result = json.loads(stdout.decode("utf-8"))
    if not isinstance(result, dict) or result.get("ok") is False:
        raise ValueError("site_preview_worker_invalid_response")
    return result


def _decode_jpeg(result: dict[str, Any]) -> bytes:
    encoded = result.get("image_base64")
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("site_preview_missing_image")
    try:
        image = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("site_preview_invalid_base64") from exc
    if len(image) > MAX_JPEG_BYTES or not image.startswith(b"\xff\xd8\xff"):
        raise ValueError("site_preview_invalid_jpeg")
    return image


def _write_snapshot(run_id: str, image: bytes) -> tuple[str, str]:
    digest = hashlib.sha256(image).hexdigest()
    run_directory = _safe_run_directory(run_id)
    run_directory.mkdir(mode=0o755, parents=True, exist_ok=True)
    filename = f"site-preview-{digest[:12]}.jpg"
    target = run_directory / filename
    temporary = run_directory / f".{filename}.{uuid.uuid4().hex}.tmp"
    with temporary.open("xb") as handle:
        handle.write(image)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return f"/static/generated/{run_directory.name}/{filename}", digest


async def capture_site_preview(
    run_id: str,
    *,
    domain: str,
    source_url: str,
    validate_url: Callable[[str], Awaitable[None]],
) -> dict[str, Any] | None:
    """Capture, cache and persist a report preview without failing the audit."""

    input_json = {
        "domain": domain,
        "source_url": source_url,
        "viewport": {
            "width": SITE_PREVIEW_WIDTH,
            "height": SITE_PREVIEW_HEIGHT,
        },
    }
    cached_artifact = await _artifact(run_id)
    if (
        cached_artifact is not None
        and cached_artifact.status == "completed"
        and cached_artifact.prompt_version == SITE_PREVIEW_VERSION
        and cached_artifact.input_json == input_json
    ):
        cached = public_site_preview(cached_artifact.output_json)
        if cached is not None:
            return cached

    try:
        await validate_url(source_url)
        await _save_artifact(
            run_id,
            status="running",
            input_json=input_json,
        )
        async with _capture_slot:
            result = await _run_worker(
                source_url,
                width=SITE_PREVIEW_WIDTH,
                height=SITE_PREVIEW_HEIGHT,
                timeout_seconds=SITE_PREVIEW_TIMEOUT_SECONDS,
            )
        image = _decode_jpeg(result)
        file_url, digest = await asyncio.to_thread(_write_snapshot, run_id, image)
        output = {
            "state": "captured",
            "file_url": file_url,
            "source_domain": domain,
            "width": SITE_PREVIEW_WIDTH,
            "height": SITE_PREVIEW_HEIGHT,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "sha256": digest,
        }
        await _save_artifact(
            run_id,
            status="completed",
            input_json=input_json,
            output_json=output,
        )
        return public_site_preview(output)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning(
            "Site preview failed for run %s: %s",
            run_id,
            type(exc).__name__,
        )
        try:
            await _save_artifact(
                run_id,
                status="failed",
                input_json=input_json,
                error_message=f"{type(exc).__name__}: {str(exc)[:800]}",
            )
        except Exception:
            logger.exception("Could not persist site preview failure for %s", run_id)
        return None
