"""Render one public website viewport in an isolated Playwright context.

The module intentionally has no application or database imports. Production
can copy it into a small, unprivileged worker environment and invoke it through
``SITE_PREVIEW_WORKER_COMMAND``. The CLI returns the JPEG as base64 JSON so the
browser process never needs write access to the application or its reports.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import ipaddress
import json
import socket
from typing import Any
from urllib.parse import urlparse

MAX_VIEWPORT_WIDTH = 1920
MAX_VIEWPORT_HEIGHT = 1440
MAX_TIMEOUT_SECONDS = 45


async def validate_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("unsafe_target")
    if parsed.username or parsed.password or parsed.port not in {None, 80, 443}:
        raise ValueError("unsafe_target")

    hostname = parsed.hostname.strip("[]").lower()
    if hostname in {"localhost", "localhost.localdomain"} or hostname.endswith(".local"):
        raise ValueError("unsafe_target")
    try:
        addresses = [ipaddress.ip_address(hostname)]
    except ValueError:
        loop = asyncio.get_running_loop()
        try:
            records = await loop.getaddrinfo(
                hostname,
                None,
                family=socket.AF_UNSPEC,
                type=socket.SOCK_STREAM,
            )
        except socket.gaierror as exc:
            raise ValueError("unsafe_target") from exc
        addresses = []
        for record in records:
            try:
                addresses.append(ipaddress.ip_address(record[4][0]))
            except ValueError:
                continue
    if not addresses or any(not address.is_global for address in addresses):
        raise ValueError("unsafe_target")


async def capture_preview(
    url: str,
    *,
    width: int = 1440,
    height: int = 900,
    timeout_seconds: int = 40,
) -> dict[str, Any]:
    """Return a deterministic first-viewport JPEG and minimal page metadata."""

    from playwright.async_api import TimeoutError as PlaywrightTimeoutError
    from playwright.async_api import async_playwright

    width = max(1024, min(MAX_VIEWPORT_WIDTH, int(width)))
    height = max(720, min(MAX_VIEWPORT_HEIGHT, int(height)))
    timeout_seconds = max(10, min(MAX_TIMEOUT_SECONDS, int(timeout_seconds)))
    timeout_ms = timeout_seconds * 1000
    await validate_public_url(url)

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            context = await browser.new_context(
                viewport={"width": width, "height": height},
                device_scale_factor=1,
                color_scheme="light",
                reduced_motion="reduce",
                locale="ru-RU",
                timezone_id="Europe/Moscow",
                service_workers="block",
                accept_downloads=False,
            )
            validation_slots = asyncio.Semaphore(16)

            async def guarded_route(route: Any) -> None:
                request = route.request
                parsed = urlparse(request.url)
                if parsed.scheme not in {"http", "https"}:
                    await route.abort("blockedbyclient")
                    return
                if request.resource_type == "media":
                    await route.abort("blockedbyclient")
                    return
                try:
                    async with validation_slots:
                        await validate_public_url(request.url)
                except (ValueError, OSError):
                    await route.abort("blockedbyclient")
                    return
                await route.continue_()

            await context.route("**/*", guarded_route)
            page = await context.new_page()
            response = await page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=timeout_ms,
            )
            await validate_public_url(page.url)
            try:
                await page.wait_for_load_state("networkidle", timeout=4500)
            except PlaywrightTimeoutError:
                pass
            try:
                await asyncio.wait_for(
                    page.evaluate(
                        """async () => {
                          if (document.fonts && document.fonts.ready) {
                            await document.fonts.ready;
                          }
                          return true;
                        }"""
                    ),
                    timeout=3.0,
                )
            except (asyncio.TimeoutError, PlaywrightTimeoutError):
                pass
            jpeg = await page.screenshot(
                type="jpeg",
                quality=84,
                full_page=False,
                animations="disabled",
                caret="hide",
                scale="css",
                timeout=min(timeout_ms, 15_000),
            )
            return {
                "image_base64": base64.b64encode(jpeg).decode("ascii"),
                "width": width,
                "height": height,
                "http_status": response.status if response is not None else None,
                "title": (await page.title())[:300],
            }
        finally:
            await browser.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--timeout", type=int, default=40)
    return parser.parse_args()


async def _main() -> None:
    args = _parse_args()
    try:
        result = await capture_preview(
            args.url,
            width=args.width,
            height=args.height,
            timeout_seconds=args.timeout,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error_class": type(exc).__name__,
                    "error": str(exc)[:500],
                },
                ensure_ascii=False,
            )
        )
        raise SystemExit(1) from exc
    print(json.dumps({"ok": True, **result}, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(_main())
