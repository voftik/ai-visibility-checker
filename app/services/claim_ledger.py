from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence, TypeAlias


CLAIM_LEDGER_VERSION = "claim-ledger-v1"
CLAIM_ID_VERSION = "source-claim-v1"
DEFAULT_FRAGMENT_UTF8_BYTES = 4_096

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _canonical_json(value: Any, *, sort_keys: bool = True) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=sort_keys,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Value is not losslessly representable as JSON: {exc}") from exc


def _json_path_member(parent: str, key: str) -> str:
    encoded = json.dumps(key, ensure_ascii=False, separators=(",", ":"))
    return f"{parent}[{encoded}]"


def _json_path_index(parent: str, index: int) -> str:
    return f"{parent}[{index}]"


def _scalar_type(value: JsonScalar) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int):
        return "number"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Non-finite numbers are not valid JSON scalars")
        return "number"
    raise ValueError(f"Unsupported JSON scalar type: {type(value).__name__}")


def _scalar_text(value: JsonScalar, scalar_type: str) -> str:
    if scalar_type == "string":
        return str(value)
    return _canonical_json(value, sort_keys=False)


def _boundary_rank(character: str) -> int:
    if character in "\n\r":
        return 4
    if character in ".!?…;:。！？":
        return 3
    if character.isspace():
        return 2
    if character in ",、，)]}»”\"'":
        return 1
    return 0


def _preferred_boundary(
    candidates: Sequence[tuple[int, int, int]],
    *,
    minimum_utf8_bytes: int,
) -> tuple[int, int] | None:
    """Choose a format-neutral cut, preferring paragraph/sentence boundaries."""

    eligible = [item for item in candidates if item[1] >= minimum_utf8_bytes]
    if not eligible:
        return None
    best_rank = max(item[2] for item in eligible)
    char_end, byte_end, _rank = max(
        (item for item in eligible if item[2] == best_rank),
        key=lambda item: item[1],
    )
    return char_end, byte_end


def _split_utf8_losslessly(
    text: str,
    *,
    target_utf8_bytes: int,
) -> list[tuple[str, int, int]]:
    """Split text without cutting a Unicode code point or changing any byte.

    ``target_utf8_bytes`` is a processing-window target, not a document limit.
    A single Unicode code point may exceed it and is kept intact. There is no
    maximum fragment count or source size in this function.
    """

    if isinstance(target_utf8_bytes, bool) or not isinstance(target_utf8_bytes, int):
        raise ValueError("target_utf8_bytes must be a positive integer")
    if target_utf8_bytes <= 0:
        raise ValueError("target_utf8_bytes must be a positive integer")
    if text == "":
        return [("", 0, 0)]

    fragments: list[tuple[str, int, int]] = []
    buffer: list[str] = []
    buffer_utf8_bytes = 0
    source_utf8_offset = 0
    candidates: list[tuple[int, int, int]] = []

    def refresh_candidates() -> None:
        nonlocal candidates
        candidates = []
        byte_end = 0
        for char_end, character in enumerate(buffer, start=1):
            byte_end += len(character.encode("utf-8"))
            rank = _boundary_rank(character)
            if rank:
                candidates.append((char_end, byte_end, rank))

    def emit(char_end: int, byte_end: int) -> None:
        nonlocal buffer, buffer_utf8_bytes, source_utf8_offset
        excerpt = "".join(buffer[:char_end])
        actual_bytes = len(excerpt.encode("utf-8"))
        if actual_bytes != byte_end:
            raise AssertionError("Internal UTF-8 boundary accounting mismatch")
        fragments.append((excerpt, source_utf8_offset, byte_end))
        source_utf8_offset += byte_end
        buffer = buffer[char_end:]
        buffer_utf8_bytes -= byte_end
        refresh_candidates()

    for character in text:
        character_utf8_bytes = len(character.encode("utf-8"))
        if buffer and buffer_utf8_bytes + character_utf8_bytes > target_utf8_bytes:
            preferred = _preferred_boundary(
                candidates,
                minimum_utf8_bytes=max(1, target_utf8_bytes * 3 // 5),
            )
            if preferred is None:
                emit(len(buffer), buffer_utf8_bytes)
            else:
                emit(*preferred)

            # A preferred cut can leave a suffix in the buffer. Flush again if
            # adding the next code point would still exceed the target.
            if buffer and buffer_utf8_bytes + character_utf8_bytes > target_utf8_bytes:
                emit(len(buffer), buffer_utf8_bytes)

        buffer.append(character)
        buffer_utf8_bytes += character_utf8_bytes
        rank = _boundary_rank(character)
        if rank:
            candidates.append((len(buffer), buffer_utf8_bytes, rank))

        # One code point can be larger than the target. Keep the code point
        # whole, then emit it immediately so forward progress is guaranteed.
        if len(buffer) == 1 and buffer_utf8_bytes > target_utf8_bytes:
            emit(1, buffer_utf8_bytes)

    if buffer:
        emit(len(buffer), buffer_utf8_bytes)

    if "".join(fragment for fragment, _offset, _length in fragments) != text:
        raise AssertionError("Internal lossless split reconstruction mismatch")
    return fragments


@dataclass(frozen=True, slots=True)
class SourceClaim:
    claim_id: str
    document_id: str
    json_path: str
    scalar_type: str
    fragment_index: int
    fragment_count: int
    source_utf8_offset: int
    source_utf8_length: int
    scalar_utf8_length: int
    scalar_sha256: str
    excerpt: str
    excerpt_sha256: str

    @property
    def source_utf8_end(self) -> int:
        return self.source_utf8_offset + self.source_utf8_length

    def descriptor(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "document_id": self.document_id,
            "json_path": self.json_path,
            "scalar_type": self.scalar_type,
            "fragment_index": self.fragment_index,
            "fragment_count": self.fragment_count,
            "source_utf8_offset": self.source_utf8_offset,
            "source_utf8_length": self.source_utf8_length,
            "scalar_utf8_length": self.scalar_utf8_length,
            "scalar_sha256": self.scalar_sha256,
            "excerpt_sha256": self.excerpt_sha256,
        }

    def coverage_reference(self) -> dict[str, str]:
        return {
            "claim_id": self.claim_id,
            "excerpt_sha256": self.excerpt_sha256,
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self.descriptor(), "excerpt": self.excerpt}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SourceClaim":
        try:
            return cls(
                claim_id=str(value["claim_id"]),
                document_id=str(value["document_id"]),
                json_path=str(value["json_path"]),
                scalar_type=str(value["scalar_type"]),
                fragment_index=int(value["fragment_index"]),
                fragment_count=int(value["fragment_count"]),
                source_utf8_offset=int(value["source_utf8_offset"]),
                source_utf8_length=int(value["source_utf8_length"]),
                scalar_utf8_length=int(value["scalar_utf8_length"]),
                scalar_sha256=str(value["scalar_sha256"]),
                excerpt=str(value["excerpt"]),
                excerpt_sha256=str(value["excerpt_sha256"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Malformed source claim: {exc}") from exc


@dataclass(frozen=True, slots=True)
class ClaimLedgerManifest:
    version: str
    document_id: str
    source_sha256: str
    source_utf8_bytes: int
    scalar_count: int
    claim_count: int
    structure: dict[str, Any]
    claim_descriptors: tuple[dict[str, Any], ...]
    manifest_sha256: str

    def payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "document_id": self.document_id,
            "source_sha256": self.source_sha256,
            "source_utf8_bytes": self.source_utf8_bytes,
            "scalar_count": self.scalar_count,
            "claim_count": self.claim_count,
            # Return a detached serialization view: callers may store or edit
            # it without mutating the otherwise frozen in-memory manifest.
            "structure": copy.deepcopy(self.structure),
            "claim_descriptors": copy.deepcopy(list(self.claim_descriptors)),
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self.payload(), "manifest_sha256": self.manifest_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ClaimLedgerManifest":
        try:
            structure = value["structure"]
            descriptors = value["claim_descriptors"]
            if not isinstance(structure, dict) or not isinstance(descriptors, list):
                raise TypeError("structure/descriptors have invalid types")
            return cls(
                version=str(value["version"]),
                document_id=str(value["document_id"]),
                source_sha256=str(value["source_sha256"]),
                source_utf8_bytes=int(value["source_utf8_bytes"]),
                scalar_count=int(value["scalar_count"]),
                claim_count=int(value["claim_count"]),
                structure=copy.deepcopy(structure),
                claim_descriptors=tuple(
                    copy.deepcopy(dict(descriptor)) for descriptor in descriptors
                ),
                manifest_sha256=str(value["manifest_sha256"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Malformed claim-ledger manifest: {exc}") from exc


@dataclass(frozen=True, slots=True)
class ClaimCoverageReport:
    expected_claim_count: int
    covered_claim_count: int
    missing_claim_ids: tuple[str, ...]

    @property
    def coverage_complete(self) -> bool:
        return not self.missing_claim_ids

    def as_dict(self) -> dict[str, Any]:
        return {
            "expected_claim_count": self.expected_claim_count,
            "covered_claim_count": self.covered_claim_count,
            "missing_claim_ids": list(self.missing_claim_ids),
            "coverage_complete": self.coverage_complete,
        }


def _claim_id_payload(
    *,
    document_id: str,
    json_path: str,
    scalar_type: str,
    fragment_index: int,
    fragment_count: int,
    source_utf8_offset: int,
    source_utf8_length: int,
    scalar_utf8_length: int,
    scalar_sha256: str,
    excerpt_sha256: str,
) -> dict[str, Any]:
    return {
        "version": CLAIM_ID_VERSION,
        "document_id": document_id,
        "json_path": json_path,
        "scalar_type": scalar_type,
        "fragment_index": fragment_index,
        "fragment_count": fragment_count,
        "source_utf8_offset": source_utf8_offset,
        "source_utf8_length": source_utf8_length,
        "scalar_utf8_length": scalar_utf8_length,
        "scalar_sha256": scalar_sha256,
        "excerpt_sha256": excerpt_sha256,
    }


def _make_claim(
    *,
    document_id: str,
    json_path: str,
    scalar_type: str,
    fragment_index: int,
    fragment_count: int,
    source_utf8_offset: int,
    source_utf8_length: int,
    scalar_utf8_length: int,
    scalar_sha256: str,
    excerpt: str,
) -> SourceClaim:
    excerpt_sha256 = _sha256_text(excerpt)
    identity_payload = _claim_id_payload(
        document_id=document_id,
        json_path=json_path,
        scalar_type=scalar_type,
        fragment_index=fragment_index,
        fragment_count=fragment_count,
        source_utf8_offset=source_utf8_offset,
        source_utf8_length=source_utf8_length,
        scalar_utf8_length=scalar_utf8_length,
        scalar_sha256=scalar_sha256,
        excerpt_sha256=excerpt_sha256,
    )
    claim_id = f"clm_{_sha256_text(_canonical_json(identity_payload))}"
    return SourceClaim(
        claim_id=claim_id,
        document_id=document_id,
        json_path=json_path,
        scalar_type=scalar_type,
        fragment_index=fragment_index,
        fragment_count=fragment_count,
        source_utf8_offset=source_utf8_offset,
        source_utf8_length=source_utf8_length,
        scalar_utf8_length=scalar_utf8_length,
        scalar_sha256=scalar_sha256,
        excerpt=excerpt,
        excerpt_sha256=excerpt_sha256,
    )


def build_claim_ledger(
    source: JsonValue,
    *,
    document_id: str,
    target_fragment_utf8_bytes: int = DEFAULT_FRAGMENT_UTF8_BYTES,
) -> tuple[list[SourceClaim], ClaimLedgerManifest]:
    """Create an exact, code-owned claim ledger for a JSON-shaped value.

    The ledger proves byte-exact source accounting and lineage. It deliberately
    does not claim that an LLM understood the meaning of every claim.
    """

    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("document_id must be a non-empty string")
    if (
        isinstance(target_fragment_utf8_bytes, bool)
        or not isinstance(target_fragment_utf8_bytes, int)
        or target_fragment_utf8_bytes <= 0
    ):
        raise ValueError("target_fragment_utf8_bytes must be a positive integer")

    claims: list[SourceClaim] = []
    scalar_count = 0

    def visit(value: JsonValue, json_path: str) -> dict[str, Any]:
        nonlocal scalar_count
        if isinstance(value, dict):
            entries: list[dict[str, Any]] = []
            for key, child in value.items():
                if not isinstance(key, str):
                    raise ValueError("Claim-ledger objects require string keys")
                entries.append(
                    {
                        "key": key,
                        "value": visit(child, _json_path_member(json_path, key)),
                    }
                )
            return {"kind": "object", "entries": entries}
        if isinstance(value, list):
            return {
                "kind": "array",
                "items": [
                    visit(child, _json_path_index(json_path, index))
                    for index, child in enumerate(value)
                ],
            }

        scalar_count += 1
        kind = _scalar_type(value)
        scalar_text = _scalar_text(value, kind)
        scalar_utf8_length = len(scalar_text.encode("utf-8"))
        scalar_sha256 = _sha256_text(scalar_text)
        fragments = _split_utf8_losslessly(
            scalar_text,
            target_utf8_bytes=target_fragment_utf8_bytes,
        )
        fragment_count = len(fragments)
        claim_ids: list[str] = []
        for fragment_index, (excerpt, offset, length) in enumerate(fragments):
            claim = _make_claim(
                document_id=document_id,
                json_path=json_path,
                scalar_type=kind,
                fragment_index=fragment_index,
                fragment_count=fragment_count,
                source_utf8_offset=offset,
                source_utf8_length=length,
                scalar_utf8_length=scalar_utf8_length,
                scalar_sha256=scalar_sha256,
                excerpt=excerpt,
            )
            claims.append(claim)
            claim_ids.append(claim.claim_id)
        return {
            "kind": "scalar",
            "json_path": json_path,
            "scalar_type": kind,
            "scalar_utf8_length": scalar_utf8_length,
            "scalar_sha256": scalar_sha256,
            "claim_ids": claim_ids,
        }

    structure = visit(source, "$")
    source_json = _canonical_json(source, sort_keys=False)
    source_utf8 = source_json.encode("utf-8")
    payload = {
        "version": CLAIM_LEDGER_VERSION,
        "document_id": document_id,
        "source_sha256": _sha256_bytes(source_utf8),
        "source_utf8_bytes": len(source_utf8),
        "scalar_count": scalar_count,
        "claim_count": len(claims),
        "structure": structure,
        "claim_descriptors": [claim.descriptor() for claim in claims],
    }
    manifest = ClaimLedgerManifest(
        version=CLAIM_LEDGER_VERSION,
        document_id=document_id,
        source_sha256=payload["source_sha256"],
        source_utf8_bytes=payload["source_utf8_bytes"],
        scalar_count=scalar_count,
        claim_count=len(claims),
        structure=structure,
        claim_descriptors=tuple(payload["claim_descriptors"]),
        manifest_sha256=_sha256_text(_canonical_json(payload)),
    )
    return claims, manifest


def _coerce_claim(value: SourceClaim | Mapping[str, Any]) -> SourceClaim:
    if isinstance(value, SourceClaim):
        return value
    if isinstance(value, Mapping):
        return SourceClaim.from_dict(value)
    raise ValueError(f"Unsupported source claim: {type(value).__name__}")


def _coerce_manifest(
    value: ClaimLedgerManifest | Mapping[str, Any],
) -> ClaimLedgerManifest:
    if isinstance(value, ClaimLedgerManifest):
        return value
    if isinstance(value, Mapping):
        return ClaimLedgerManifest.from_dict(value)
    raise ValueError(f"Unsupported claim-ledger manifest: {type(value).__name__}")


def _validate_claim_identity(claim: SourceClaim) -> None:
    if claim.scalar_type not in {"string", "number", "boolean", "null"}:
        raise ValueError(f"Claim {claim.claim_id} has an invalid scalar type")
    if claim.fragment_count <= 0:
        raise ValueError(f"Claim {claim.claim_id} has an invalid fragment count")
    if not 0 <= claim.fragment_index < claim.fragment_count:
        raise ValueError(f"Claim {claim.claim_id} has an invalid fragment index")
    if claim.source_utf8_offset < 0 or claim.source_utf8_length < 0:
        raise ValueError(f"Claim {claim.claim_id} has invalid UTF-8 offsets")
    if claim.scalar_utf8_length < 0:
        raise ValueError(f"Claim {claim.claim_id} has invalid scalar length")
    actual_length = len(claim.excerpt.encode("utf-8"))
    if actual_length != claim.source_utf8_length:
        raise ValueError(f"Claim {claim.claim_id} UTF-8 length was tampered")
    actual_excerpt_sha256 = _sha256_text(claim.excerpt)
    if actual_excerpt_sha256 != claim.excerpt_sha256:
        raise ValueError(f"Claim {claim.claim_id} excerpt digest mismatch")
    expected_payload = _claim_id_payload(
        document_id=claim.document_id,
        json_path=claim.json_path,
        scalar_type=claim.scalar_type,
        fragment_index=claim.fragment_index,
        fragment_count=claim.fragment_count,
        source_utf8_offset=claim.source_utf8_offset,
        source_utf8_length=claim.source_utf8_length,
        scalar_utf8_length=claim.scalar_utf8_length,
        scalar_sha256=claim.scalar_sha256,
        excerpt_sha256=claim.excerpt_sha256,
    )
    expected_id = f"clm_{_sha256_text(_canonical_json(expected_payload))}"
    if claim.claim_id != expected_id:
        raise ValueError(f"Claim {claim.claim_id} identity mismatch")


def reconstruct_claim_ledger(
    claims: Sequence[SourceClaim | Mapping[str, Any]],
    manifest: ClaimLedgerManifest | Mapping[str, Any],
) -> JsonValue:
    """Validate and reconstruct the original value, failing closed on drift."""

    normalized_manifest = _coerce_manifest(manifest)
    if normalized_manifest.version != CLAIM_LEDGER_VERSION:
        raise ValueError(
            f"Unsupported claim-ledger version: {normalized_manifest.version}"
        )
    expected_manifest_sha256 = _sha256_text(
        _canonical_json(normalized_manifest.payload())
    )
    if normalized_manifest.manifest_sha256 != expected_manifest_sha256:
        raise ValueError("Claim-ledger manifest digest mismatch")

    normalized_claims = [_coerce_claim(claim) for claim in claims]
    by_id: dict[str, SourceClaim] = {}
    for claim in normalized_claims:
        _validate_claim_identity(claim)
        if claim.claim_id in by_id:
            raise ValueError(f"Duplicate claim: {claim.claim_id}")
        by_id[claim.claim_id] = claim

    if len(normalized_claims) != normalized_manifest.claim_count:
        raise ValueError("Claim count does not match the manifest")
    descriptor_by_id: dict[str, dict[str, Any]] = {}
    for raw_descriptor in normalized_manifest.claim_descriptors:
        descriptor = dict(raw_descriptor)
        claim_id = descriptor.get("claim_id")
        if not isinstance(claim_id, str) or not claim_id:
            raise ValueError("Manifest contains a malformed claim descriptor")
        if claim_id in descriptor_by_id:
            raise ValueError(f"Manifest contains duplicate claim: {claim_id}")
        descriptor_by_id[claim_id] = descriptor

    expected_ids = set(descriptor_by_id)
    actual_ids = set(by_id)
    unknown_ids = sorted(actual_ids - expected_ids)
    missing_ids = sorted(expected_ids - actual_ids)
    if unknown_ids:
        raise ValueError(f"Unknown claims: {', '.join(unknown_ids)}")
    if missing_ids:
        raise ValueError(f"Missing claims: {', '.join(missing_ids)}")
    for claim_id, descriptor in descriptor_by_id.items():
        if by_id[claim_id].descriptor() != descriptor:
            raise ValueError(f"Claim descriptor mismatch: {claim_id}")

    used_claim_ids: set[str] = set()
    scalar_count = 0

    def reconstruct_node(node: Any, json_path: str) -> JsonValue:
        nonlocal scalar_count
        if not isinstance(node, dict):
            raise ValueError(f"Malformed structure node at {json_path}")
        kind = node.get("kind")
        if kind == "object":
            entries = node.get("entries")
            if not isinstance(entries, list):
                raise ValueError(f"Malformed object node at {json_path}")
            result: dict[str, JsonValue] = {}
            for entry in entries:
                if not isinstance(entry, dict) or set(entry) != {"key", "value"}:
                    raise ValueError(f"Malformed object entry at {json_path}")
                key = entry["key"]
                if not isinstance(key, str):
                    raise ValueError(f"Non-string object key at {json_path}")
                if key in result:
                    raise ValueError(f"Duplicate object key at {json_path}: {key}")
                result[key] = reconstruct_node(
                    entry["value"],
                    _json_path_member(json_path, key),
                )
            return result
        if kind == "array":
            items = node.get("items")
            if not isinstance(items, list):
                raise ValueError(f"Malformed array node at {json_path}")
            return [
                reconstruct_node(item, _json_path_index(json_path, index))
                for index, item in enumerate(items)
            ]
        if kind != "scalar":
            raise ValueError(f"Unknown structure node kind at {json_path}: {kind}")

        scalar_count += 1
        if node.get("json_path") != json_path:
            raise ValueError(f"Scalar JSON path mismatch at {json_path}")
        scalar_type = node.get("scalar_type")
        claim_ids = node.get("claim_ids")
        if scalar_type not in {"string", "number", "boolean", "null"}:
            raise ValueError(f"Invalid scalar type at {json_path}")
        if not isinstance(claim_ids, list) or not claim_ids:
            raise ValueError(f"Scalar at {json_path} has no claims")
        if len(set(claim_ids)) != len(claim_ids):
            raise ValueError(f"Scalar at {json_path} repeats a claim")

        scalar_claims: list[SourceClaim] = []
        expected_offset = 0
        for fragment_index, claim_id in enumerate(claim_ids):
            if not isinstance(claim_id, str) or claim_id not in by_id:
                raise ValueError(f"Unknown claim in scalar {json_path}: {claim_id}")
            if claim_id in used_claim_ids:
                raise ValueError(f"Claim is referenced more than once: {claim_id}")
            claim = by_id[claim_id]
            if claim.document_id != normalized_manifest.document_id:
                raise ValueError(f"Claim belongs to another document: {claim_id}")
            if claim.json_path != json_path or claim.scalar_type != scalar_type:
                raise ValueError(f"Claim lineage mismatch: {claim_id}")
            if claim.fragment_index != fragment_index:
                raise ValueError(f"Claim fragment order mismatch: {claim_id}")
            if claim.fragment_count != len(claim_ids):
                raise ValueError(f"Claim fragment count mismatch: {claim_id}")
            if claim.source_utf8_offset != expected_offset:
                raise ValueError(f"Claim UTF-8 offset gap or overlap: {claim_id}")
            expected_offset = claim.source_utf8_end
            scalar_claims.append(claim)
            used_claim_ids.add(claim_id)

        scalar_text = "".join(claim.excerpt for claim in scalar_claims)
        scalar_utf8_length = len(scalar_text.encode("utf-8"))
        scalar_sha256 = _sha256_text(scalar_text)
        if expected_offset != scalar_utf8_length:
            raise ValueError(f"Scalar UTF-8 coverage mismatch at {json_path}")
        if node.get("scalar_utf8_length") != scalar_utf8_length:
            raise ValueError(f"Scalar UTF-8 length mismatch at {json_path}")
        if node.get("scalar_sha256") != scalar_sha256:
            raise ValueError(f"Scalar digest mismatch at {json_path}")
        for claim in scalar_claims:
            if (
                claim.scalar_utf8_length != scalar_utf8_length
                or claim.scalar_sha256 != scalar_sha256
            ):
                raise ValueError(f"Claim scalar digest mismatch: {claim.claim_id}")

        if scalar_type == "string":
            return scalar_text
        try:
            parsed = json.loads(scalar_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid scalar JSON at {json_path}: {exc}") from exc
        if scalar_type == "null" and parsed is not None:
            raise ValueError(f"Scalar type mismatch at {json_path}")
        if scalar_type == "boolean" and not isinstance(parsed, bool):
            raise ValueError(f"Scalar type mismatch at {json_path}")
        if scalar_type == "number" and (
            isinstance(parsed, bool) or not isinstance(parsed, (int, float))
        ):
            raise ValueError(f"Scalar type mismatch at {json_path}")
        return parsed

    reconstructed = reconstruct_node(normalized_manifest.structure, "$")
    if used_claim_ids != actual_ids:
        unused = sorted(actual_ids - used_claim_ids)
        raise ValueError(f"Claims are not referenced by the structure: {', '.join(unused)}")
    if scalar_count != normalized_manifest.scalar_count:
        raise ValueError("Scalar count does not match the manifest")
    reconstructed_json = _canonical_json(reconstructed, sort_keys=False)
    reconstructed_utf8 = reconstructed_json.encode("utf-8")
    if len(reconstructed_utf8) != normalized_manifest.source_utf8_bytes:
        raise ValueError("Reconstructed source UTF-8 length mismatch")
    if _sha256_bytes(reconstructed_utf8) != normalized_manifest.source_sha256:
        raise ValueError("Reconstructed source digest mismatch")
    return reconstructed


def validate_claim_ledger(
    claims: Sequence[SourceClaim | Mapping[str, Any]],
    manifest: ClaimLedgerManifest | Mapping[str, Any],
) -> None:
    """Validate a ledger without treating byte coverage as semantic quality."""

    reconstruct_claim_ledger(claims, manifest)


def claim_coverage_references(
    claims: Iterable[SourceClaim | Mapping[str, Any]],
) -> list[dict[str, str]]:
    normalized: list[SourceClaim] = []
    seen: set[str] = set()
    for raw_claim in claims:
        claim = _coerce_claim(raw_claim)
        _validate_claim_identity(claim)
        if claim.claim_id in seen:
            raise ValueError(f"Duplicate claim: {claim.claim_id}")
        seen.add(claim.claim_id)
        normalized.append(claim)
    return [claim.coverage_reference() for claim in normalized]


def validate_claim_coverage(
    claims: Sequence[SourceClaim | Mapping[str, Any]],
    coverage: Iterable[Mapping[str, Any]],
    *,
    manifest: ClaimLedgerManifest | Mapping[str, Any] | None = None,
    require_complete: bool = True,
) -> ClaimCoverageReport:
    """Validate code-owned claim references returned by a map/reduce layer.

    Complete reference coverage means every exact source claim was accounted
    for. It does not assert that the model interpreted each claim correctly.
    """

    normalized_claims = [_coerce_claim(claim) for claim in claims]
    if manifest is not None:
        validate_claim_ledger(normalized_claims, manifest)
    expected: dict[str, SourceClaim] = {}
    for claim in normalized_claims:
        _validate_claim_identity(claim)
        if claim.claim_id in expected:
            raise ValueError(f"Duplicate claim: {claim.claim_id}")
        expected[claim.claim_id] = claim

    seen: set[str] = set()
    for raw_reference in coverage:
        if not isinstance(raw_reference, Mapping):
            raise ValueError("Coverage references must be objects")
        claim_id = raw_reference.get("claim_id")
        excerpt_sha256 = raw_reference.get("excerpt_sha256")
        if not isinstance(claim_id, str) or not claim_id:
            raise ValueError("Coverage reference has no valid claim_id")
        if claim_id in seen:
            raise ValueError(f"Duplicate coverage claim: {claim_id}")
        if claim_id not in expected:
            raise ValueError(f"Unknown coverage claim: {claim_id}")
        if excerpt_sha256 != expected[claim_id].excerpt_sha256:
            raise ValueError(f"Coverage digest mismatch: {claim_id}")
        seen.add(claim_id)

    missing = tuple(
        claim.claim_id for claim in normalized_claims if claim.claim_id not in seen
    )
    if require_complete and missing:
        raise ValueError(f"Incomplete claim coverage; missing {len(missing)} claims")
    return ClaimCoverageReport(
        expected_claim_count=len(normalized_claims),
        covered_claim_count=len(seen),
        missing_claim_ids=missing,
    )


__all__ = [
    "CLAIM_LEDGER_VERSION",
    "DEFAULT_FRAGMENT_UTF8_BYTES",
    "ClaimCoverageReport",
    "ClaimLedgerManifest",
    "JsonValue",
    "SourceClaim",
    "build_claim_ledger",
    "claim_coverage_references",
    "reconstruct_claim_ledger",
    "validate_claim_coverage",
    "validate_claim_ledger",
]
