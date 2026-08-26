from __future__ import annotations

import base64
import json
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.services import site_preview
from app.services.site_preview_worker import (
    WORKER_PROTOCOL_VERSION,
    validate_public_url,
)


class SitePreviewWorkerSafetyTests(unittest.IsolatedAsyncioTestCase):
    def test_worker_advertises_versioned_navigation_protocol(self) -> None:
        worker = Path(site_preview.__file__).with_name("site_preview_worker.py")
        completed = subprocess.run(
            [
                sys.executable,
                str(worker),
                "--protocol-version",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["worker_protocol_version"], WORKER_PROTOCOL_VERSION)
        self.assertIn("navigation", payload["modes"])

    async def test_rejects_local_private_and_non_http_targets(self) -> None:
        for url in (
            "http://127.0.0.1/",
            "http://10.0.0.1/",
            "http://169.254.169.254/latest/meta-data/",
            "http://[::1]/",
            "file:///etc/passwd",
            "https://user:password@example.com/",
            "https://example.com:8443/",
        ):
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    await validate_public_url(url)

    async def test_accepts_a_public_https_address(self) -> None:
        await validate_public_url("https://93.184.216.34/")


class SitePreviewArtifactTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.static_dir = Path(self.temporary.name) / "static"
        self.generated_dir = self.static_dir / "generated"
        self.static_patch = patch.object(site_preview, "STATIC_DIR", self.static_dir)
        self.generated_patch = patch.object(
            site_preview,
            "GENERATED_DIR",
            self.generated_dir,
        )
        self.static_patch.start()
        self.generated_patch.start()

    def tearDown(self) -> None:
        self.generated_patch.stop()
        self.static_patch.stop()
        self.temporary.cleanup()

    def test_snapshot_path_is_hashed_and_confined_to_the_run(self) -> None:
        run_id = str(uuid.uuid4())
        file_url, digest = site_preview._write_snapshot(
            run_id,
            b"\xff\xd8\xff" + b"report-preview",
        )
        self.assertRegex(
            file_url,
            rf"^/static/generated/{run_id}/site-preview-[0-9a-f]{{12}}\.jpg$",
        )
        self.assertEqual(len(digest), 64)
        self.assertTrue(site_preview._path_for_file_url(file_url).is_file())
        with self.assertRaises(ValueError):
            site_preview._safe_run_directory("../../outside")

    def test_public_preview_binds_run_filename_and_file_bytes(self) -> None:
        run_id = str(uuid.uuid4())
        image = b"\xff\xd8\xff" + b"immutable-preview"
        file_url, digest = site_preview._write_snapshot(run_id, image)
        payload = {
            "file_url": file_url,
            "source_domain": "example.com",
            "width": 1440,
            "height": 900,
            "captured_at": "2026-08-26T00:00:00+00:00",
            "sha256": digest,
        }

        self.assertIsNotNone(
            site_preview.public_site_preview(payload, run_id=run_id)
        )
        receipt = site_preview.site_preview_asset_receipt(run_id, payload)
        self.assertEqual(receipt["image_sha256"], digest)
        self.assertEqual(receipt["run_id"], run_id)
        self.assertEqual(receipt["image_bytes"], len(image))
        self.assertRegex(receipt["receipt_sha256"], r"^[0-9a-f]{64}$")
        self.assertIsNone(
            site_preview.public_site_preview(payload, run_id=str(uuid.uuid4()))
        )
        self.assertIsNone(
            site_preview.public_site_preview(
                {**payload, "sha256": "0" * 64},
                run_id=run_id,
            )
        )

        site_preview._path_for_file_url(file_url).write_bytes(
            b"\xff\xd8\xff" + b"mutated-preview"
        )
        self.assertIsNone(
            site_preview.public_site_preview(payload, run_id=run_id)
        )

    async def test_corrupted_cached_artifact_is_not_reused(self) -> None:
        run_id = str(uuid.uuid4())
        file_url, digest = site_preview._write_snapshot(
            run_id,
            b"\xff\xd8\xff" + b"cached",
        )
        site_preview._path_for_file_url(file_url).write_bytes(
            b"\xff\xd8\xff" + b"corrupted",
        )
        input_json = {
            "domain": "example.com",
            "source_url": "https://example.com/",
            "viewport": {"width": 1440, "height": 900},
        }
        artifact = SimpleNamespace(
            status="completed",
            prompt_version=site_preview.SITE_PREVIEW_VERSION,
            input_json=input_json,
            output_json={
                "file_url": file_url,
                "source_domain": "example.com",
                "width": 1440,
                "height": 900,
                "captured_at": "2026-07-30T00:00:00+00:00",
                "sha256": digest,
            },
        )
        fresh = b"\xff\xd8\xff" + b"fresh"
        worker_result = {
            "ok": True,
            "image_base64": base64.b64encode(fresh).decode("ascii"),
            "width": 1440,
            "height": 900,
        }
        with (
            patch.object(site_preview, "_artifact", new=AsyncMock(return_value=artifact)),
            patch.object(site_preview, "_save_artifact", new=AsyncMock()),
            patch.object(
                site_preview,
                "_run_worker",
                new=AsyncMock(return_value=worker_result),
            ) as worker,
        ):
            result = await site_preview.capture_site_preview(
                run_id,
                domain="example.com",
                source_url="https://example.com/",
                validate_url=AsyncMock(),
            )

        worker.assert_awaited_once()
        self.assertEqual(result["sha256"], site_preview.hashlib.sha256(fresh).hexdigest())

    async def test_cached_preview_from_another_domain_is_not_reused(self) -> None:
        run_id = str(uuid.uuid4())
        file_url, digest = site_preview._write_snapshot(
            run_id,
            b"\xff\xd8\xff" + b"cached",
        )
        artifact = SimpleNamespace(
            status="completed",
            prompt_version=site_preview.SITE_PREVIEW_VERSION,
            input_json={
                "domain": "example.com",
                "source_url": "https://example.com/",
                "viewport": {"width": 1440, "height": 900},
            },
            output_json={
                "file_url": file_url,
                "source_domain": "other.example",
                "width": 1440,
                "height": 900,
                "captured_at": "2026-07-30T00:00:00+00:00",
                "sha256": digest,
            },
        )
        fresh = b"\xff\xd8\xff" + b"fresh"
        with (
            patch.object(site_preview, "_artifact", new=AsyncMock(return_value=artifact)),
            patch.object(site_preview, "_save_artifact", new=AsyncMock()),
            patch.object(
                site_preview,
                "_run_worker",
                new=AsyncMock(
                    return_value={
                        "ok": True,
                        "image_base64": base64.b64encode(fresh).decode("ascii"),
                        "width": 1440,
                        "height": 900,
                    }
                ),
            ) as worker,
        ):
            result = await site_preview.capture_site_preview(
                run_id,
                domain="example.com",
                source_url="https://example.com/",
                validate_url=AsyncMock(),
            )

        worker.assert_awaited_once()
        self.assertEqual(result["source_domain"], "example.com")

    async def test_completed_cached_artifact_does_not_launch_browser(self) -> None:
        run_id = str(uuid.uuid4())
        image = b"\xff\xd8\xff" + b"cached"
        file_url, digest = site_preview._write_snapshot(run_id, image)
        input_json = {
            "domain": "example.com",
            "source_url": "https://example.com/",
            "viewport": {"width": 1440, "height": 900},
        }
        artifact = SimpleNamespace(
            status="completed",
            prompt_version=site_preview.SITE_PREVIEW_VERSION,
            input_json=input_json,
            output_json={
                "file_url": file_url,
                "source_domain": "example.com",
                "width": 1440,
                "height": 900,
                "captured_at": "2026-07-30T00:00:00+00:00",
                "sha256": digest,
            },
        )
        with (
            patch.object(site_preview, "_artifact", new=AsyncMock(return_value=artifact)),
            patch.object(site_preview, "_run_worker", new=AsyncMock()) as worker,
        ):
            result = await site_preview.capture_site_preview(
                run_id,
                domain="example.com",
                source_url="https://example.com/",
                validate_url=AsyncMock(),
            )
        self.assertEqual(result["file_url"], file_url)
        worker.assert_not_awaited()

    async def test_browser_failure_is_saved_but_does_not_escape(self) -> None:
        save = AsyncMock()
        with (
            patch.object(site_preview, "_artifact", new=AsyncMock(return_value=None)),
            patch.object(site_preview, "_save_artifact", new=save),
            patch.object(
                site_preview,
                "_run_worker",
                new=AsyncMock(side_effect=TimeoutError("slow page")),
            ),
        ):
            result = await site_preview.capture_site_preview(
                str(uuid.uuid4()),
                domain="example.com",
                source_url="https://example.com/",
                validate_url=AsyncMock(),
            )
        self.assertIsNone(result)
        self.assertEqual(save.await_args_list[-1].kwargs["status"], "failed")

    async def test_success_writes_public_metadata_without_image_payload(self) -> None:
        run_id = str(uuid.uuid4())
        jpeg = b"\xff\xd8\xff" + b"fresh"
        save = AsyncMock()
        worker_result = {
            "ok": True,
            "image_base64": base64.b64encode(jpeg).decode("ascii"),
            "width": 1440,
            "height": 900,
        }
        with (
            patch.object(site_preview, "_artifact", new=AsyncMock(return_value=None)),
            patch.object(site_preview, "_save_artifact", new=save),
            patch.object(
                site_preview,
                "_run_worker",
                new=AsyncMock(return_value=worker_result),
            ),
        ):
            result = await site_preview.capture_site_preview(
                run_id,
                domain="example.com",
                source_url="https://example.com/",
                validate_url=AsyncMock(),
            )
        self.assertIsNotNone(result)
        self.assertNotIn("image_base64", result)
        self.assertEqual(result["source_domain"], "example.com")
        self.assertTrue(site_preview._path_for_file_url(result["file_url"]).is_file())
        self.assertEqual(save.await_args_list[-1].kwargs["status"], "completed")


if __name__ == "__main__":
    unittest.main()
