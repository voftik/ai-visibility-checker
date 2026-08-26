"""Versioned code-owned copy shared by the report renderer and browser UI.

Model-authored report prose has its own editorial receipts.  This registry
covers the reader-facing strings that code can render without a model result,
including Markdown labels and technical fallback explanations.  The browser
loads the same file that this module verifies, so a UI wording change cannot
silently bypass the live-Russian gate or a publication manifest.

Raw answers, source quotes and citations do not belong here.  They remain
literal evidence and keep their path-aware exclusions in
``live_russian_policy``.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.services.live_russian_policy import (
    LIVE_RUSSIAN_POLICY_MANIFEST,
    lint_reader_copy_tree,
    trusted_live_russian_policy_manifest,
)

READER_COPY_REGISTRY_VERSION = "aiv-reader-copy-registry-ru-v1"
READER_COPY_REGISTRY_ASSET = "reader-copy-registry.ru.v2026-08-26.js"
READER_COPY_REGISTRY_FILE_SHA256 = (
    "be87f05a50550c57cc9c1b89adb11947014e5db26f7ea5127a63044538007435"
)
READER_COPY_REGISTRY_DOCUMENT_SHA256 = (
    "cbdc00d32e81166d0fbcca96d1e50e659d820f1980098fd5cef46c206a159425"
)

_REGISTRY_PREFIX = "window.AIV_READER_COPY_REGISTRY = Object.freeze("
_REGISTRY_SUFFIX = ");\n"


@dataclass(frozen=True, slots=True)
class ReaderCopyRegistryManifest:
    """Stable identity embedded into every reader-copy publication manifest."""

    version: str
    asset: str
    file_sha256: str
    document_sha256: str
    live_russian_policy_items: tuple[tuple[str, str], ...]
    language: str = "ru"

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "asset": f"static/{self.asset}",
            "file_sha256": self.file_sha256,
            "document_sha256": self.document_sha256,
            "language": self.language,
            "live_russian_policy": dict(self.live_russian_policy_items),
        }


READER_COPY_REGISTRY_MANIFEST = ReaderCopyRegistryManifest(
    version=READER_COPY_REGISTRY_VERSION,
    asset=READER_COPY_REGISTRY_ASSET,
    file_sha256=READER_COPY_REGISTRY_FILE_SHA256,
    document_sha256=READER_COPY_REGISTRY_DOCUMENT_SHA256,
    live_russian_policy_items=tuple(
        sorted(LIVE_RUSSIAN_POLICY_MANIFEST.as_dict().items())
    ),
)

# Append the previous descriptor here before bumping the current registry.  A
# historical publication remains readable only while both this code-owned
# descriptor and its exact versioned browser asset are retained.
_READER_COPY_REGISTRY_RU_V1 = ReaderCopyRegistryManifest(
    version="aiv-reader-copy-registry-ru-v1",
    asset="reader-copy-registry.ru.v2026-08-26.js",
    file_sha256="be87f05a50550c57cc9c1b89adb11947014e5db26f7ea5127a63044538007435",
    document_sha256=(
        "cbdc00d32e81166d0fbcca96d1e50e659d820f1980098fd5cef46c206a159425"
    ),
    live_russian_policy_items=(
        ("language", "ru"),
        ("sha256", "0cd7bbc6cdb006331b3df3c414cc0cdb9bc9860dfa2706b098fe610778392d84"),
        ("snapshot", "app/policies/live_russian_ru.v2026-07-29.md"),
        ("source_date", "2026-07-29"),
        ("version", "live-russian-2026-07-29.1"),
    ),
)
TRUSTED_READER_COPY_REGISTRY_MANIFESTS: tuple[ReaderCopyRegistryManifest, ...] = tuple(
    dict.fromkeys(
        (
            _READER_COPY_REGISTRY_RU_V1,
            READER_COPY_REGISTRY_MANIFEST,
        )
    )
)


def _registry_path() -> Path:
    return Path(__file__).resolve().parents[2] / "static" / READER_COPY_REGISTRY_ASSET


def _load_registry_document_from_path(
    path: Path,
    manifest: ReaderCopyRegistryManifest,
) -> dict[str, Any]:
    """Load one retained version without consulting mutable current constants."""

    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"Reader-copy registry is unavailable: {path}") from exc
    actual_file_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_file_sha256 != manifest.file_sha256:
        raise RuntimeError(
            "Reader-copy registry file checksum mismatch: "
            f"expected {manifest.file_sha256}, got {actual_file_sha256}"
        )
    try:
        source = payload.decode("utf-8")
    except UnicodeDecodeError as exc:  # pragma: no cover - checksum pins UTF-8
        raise RuntimeError("Reader-copy registry is not valid UTF-8") from exc
    if not source.startswith(_REGISTRY_PREFIX) or not source.endswith(_REGISTRY_SUFFIX):
        raise RuntimeError("Reader-copy registry has an invalid browser envelope")
    serialized = source[len(_REGISTRY_PREFIX) : -len(_REGISTRY_SUFFIX)]
    try:
        document = json.loads(serialized)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Reader-copy registry payload is not valid JSON") from exc
    if not isinstance(document, dict):
        raise TypeError("Reader-copy registry payload must be an object")
    if document.get("version") != manifest.version:
        raise RuntimeError("Reader-copy registry version mismatch")
    if document.get("language") != manifest.language:
        raise RuntimeError("Reader-copy registry language mismatch")
    copy_tree = document.get("copy")
    _validate_copy_tree(copy_tree)
    actual_document_sha256 = _stable_json_sha256(document)
    if actual_document_sha256 != manifest.document_sha256:
        raise RuntimeError(
            "Reader-copy registry document checksum mismatch: "
            f"expected {manifest.document_sha256}, got {actual_document_sha256}"
        )
    return document


def _stable_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_copy_tree(value: Any, *, path: str = "copy") -> None:
    if not isinstance(value, dict) or not value:
        raise RuntimeError(f"Reader-copy registry {path} must be a non-empty object")
    for key, child in value.items():
        if not isinstance(key, str) or not key:
            raise RuntimeError(f"Reader-copy registry {path} has an invalid key")
        child_path = f"{path}.{key}"
        if isinstance(child, dict):
            _validate_copy_tree(child, path=child_path)
        elif not isinstance(child, str) or not child.strip():
            raise RuntimeError(
                f"Reader-copy registry leaf {child_path} must be non-empty text"
            )


@lru_cache(maxsize=1)
def load_reader_copy_registry() -> dict[str, Any]:
    """Load the browser asset and reject byte or semantic drift."""

    return _load_registry_document_from_path(
        _registry_path(),
        READER_COPY_REGISTRY_MANIFEST,
    )


def trusted_reader_copy_registry_manifest(
    value: Any,
) -> ReaderCopyRegistryManifest | None:
    """Resolve one archived descriptor and re-hash its retained browser asset."""

    if not isinstance(value, Mapping):
        return None
    descriptor = dict(value)
    repository_root = Path(__file__).resolve().parents[2]
    static_root = (repository_root / "static").resolve()
    for candidate in TRUSTED_READER_COPY_REGISTRY_MANIFESTS:
        candidate_descriptor = candidate.as_dict()
        if descriptor != candidate_descriptor:
            continue
        if (
            trusted_live_russian_policy_manifest(
                candidate_descriptor.get("live_russian_policy")
            )
            is None
        ):
            return None
        asset_path = (static_root / candidate.asset).resolve()
        if not asset_path.is_relative_to(static_root) or not asset_path.is_file():
            return None
        try:
            _load_registry_document_from_path(asset_path, candidate)
        except (OSError, RuntimeError, TypeError, ValueError):
            return None
        return candidate
    return None


def reader_copy_registry_document() -> dict[str, Any]:
    """Return a detached document suitable for a signed publication input."""

    return copy.deepcopy(load_reader_copy_registry())


def reader_copy_value(path: str) -> str:
    """Read one required string by a stable dotted path."""

    value: Any = load_reader_copy_registry().get("copy")
    for segment in path.split("."):
        if not segment or not isinstance(value, dict) or segment not in value:
            raise KeyError(f"Unknown reader-copy registry path: {path}")
        value = value[segment]
    if not isinstance(value, str) or not value.strip():
        raise KeyError(f"Reader-copy registry path is not text: {path}")
    return value


def assert_reader_copy_registry_integrity() -> ReaderCopyRegistryManifest:
    """Run exact-byte validation and the complete deterministic Russian lint."""

    copy_tree = load_reader_copy_registry()["copy"]
    lint = lint_reader_copy_tree(
        copy_tree,
        excluded_subtrees=frozenset(),
        excluded_keys=frozenset(),
    )
    if lint.skipped_paths or lint.omitted_issue_count or lint.issues:
        issue_codes = ", ".join(issue.code for issue in lint.issues)
        raise RuntimeError(
            "Reader-copy registry failed the live-Russian gate: "
            + (issue_codes or "lint output is incomplete")
        )
    return READER_COPY_REGISTRY_MANIFEST


__all__ = [
    "READER_COPY_REGISTRY_ASSET",
    "READER_COPY_REGISTRY_DOCUMENT_SHA256",
    "READER_COPY_REGISTRY_FILE_SHA256",
    "READER_COPY_REGISTRY_MANIFEST",
    "READER_COPY_REGISTRY_VERSION",
    "TRUSTED_READER_COPY_REGISTRY_MANIFESTS",
    "ReaderCopyRegistryManifest",
    "assert_reader_copy_registry_integrity",
    "load_reader_copy_registry",
    "reader_copy_registry_document",
    "reader_copy_value",
    "trusted_reader_copy_registry_manifest",
]
