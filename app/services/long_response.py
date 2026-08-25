"""Lossless primitives for long LLM inputs and outputs.

The limits in this module are *window sizes*, never corpus limits.  A document
may produce any number of units.  Every partition carries byte/character
offsets and hashes, and reconstruction is verified before a caller is allowed
to use the units.

The pipeline deliberately distinguishes three response modes:

``atomic``
    One provider turn is the observation.  Panel answers and authoritative
    verdicts live here; a second generation must never be concatenated to the
    first one.
``partitioned``
    Code owns the complete set of unit ids and reducers must account for every
    unit.  This is the mode for extraction and annotation.
``continuable_document``
    Narrative-only documents may be continued with exact boundary overlap.
    Their assembled text is not itself a metric source.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Iterable


LONG_RESPONSE_HARNESS_VERSION = "aiv-long-response-harness-v2"
LOSSLESS_PARTITION_VERSION = "aiv-lossless-partition-v2"
DEFAULT_CONTEXT_OVERLAP_CHARS = 1_024
DEFAULT_STRUCTURED_CONTINUATION_OVERLAP_CHARS = 512
STRUCTURED_CONTINUATION_VERSION = "aiv-structured-continuation-v2"


class ResponseMode(StrEnum):
    ATOMIC = "atomic"
    PARTITIONED = "partitioned"
    CONTINUABLE_DOCUMENT = "continuable_document"


@dataclass(frozen=True)
class TextUnit:
    unit_id: str
    index: int
    start_char: int
    end_char: int
    text: str
    sha256: str
    utf8_bytes: int
    context_start_char: int
    context_end_char: int
    context_text: str
    context_sha256: str
    context_utf8_bytes: int
    core_start_in_context: int
    core_end_in_context: int

    def metadata(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("text")
        value.pop("context_text")
        return value


@dataclass(frozen=True)
class TextManifest:
    version: str
    document_id: str
    source_sha256: str
    source_chars: int
    source_utf8_bytes: int
    unit_count: int
    units: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        # Database JSON columns and wire JSON normalize tuples to arrays.  A
        # manifest must therefore be born JSON-canonical; otherwise a cached
        # artifact can never equal the freshly recomputed payload after a
        # crash/restart even though every byte is identical.
        return {
            "version": self.version,
            "document_id": self.document_id,
            "source_sha256": self.source_sha256,
            "source_chars": self.source_chars,
            "source_utf8_bytes": self.source_utf8_bytes,
            "unit_count": self.unit_count,
            "units": [dict(unit) for unit in self.units],
        }


def json_prefix_cursor(text: str) -> dict[str, Any]:
    """Describe the lexical state at the end of a possibly truncated JSON value.

    This is deliberately not a permissive JSON parser.  It catches an
    impossible bracket sequence before another paid call and gives the model a
    compact, code-owned cursor.  The assembled document still has to pass a
    normal JSON parser and the caller's Draft 2020-12 schema before use.
    """

    stack: list[str] = []
    in_string = False
    escape_pending = False
    for position, character in enumerate(text):
        if in_string:
            if escape_pending:
                escape_pending = False
            elif character == "\\":
                escape_pending = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            stack.append("object")
        elif character == "[":
            stack.append("array")
        elif character in "}]":
            expected = "object" if character == "}" else "array"
            if not stack or stack[-1] != expected:
                raise ValueError(
                    "JSON prefix has an impossible closing bracket "
                    f"at character {position}"
                )
            stack.pop()
    return {
        "chars": len(text),
        "utf8_bytes": len(text.encode("utf-8")),
        "sha256": text_sha256(text),
        "container_stack": list(stack),
        "in_string": in_string,
        "escape_pending": escape_pending,
    }


@dataclass
class StructuredContinuationLedger:
    """Code-owned audit ledger for literal structured-document continuation.

    Provider fragments never choose their sequence numbers or boundaries.
    ``append`` accepts a fragment only when it starts with the exact suffix
    issued by ``cursor`` and adds at least one new character.  This prevents a
    retry from silently becoming a second independent generation.
    """

    document_id: str
    text: str
    overlap_chars: int = DEFAULT_STRUCTURED_CONTINUATION_OVERLAP_CHARS
    parts: list[dict[str, Any]] = field(default_factory=list)
    _seen_response_hashes: set[str] = field(default_factory=set, repr=False)

    def __post_init__(self) -> None:
        if not self.document_id:
            raise ValueError("Continuation document_id must not be empty")
        if (
            isinstance(self.overlap_chars, bool)
            or not isinstance(self.overlap_chars, int)
            or self.overlap_chars <= 0
        ):
            raise ValueError("overlap_chars must be a positive integer")
        # Validate the provider prefix while it is still inside the caller's
        # initial-prefix error boundary.  Deferring this until ``cursor()``
        # used to leak a bare ValueError (and therefore lose raw/usage audit)
        # for prefixes such as ``{\"items\":]``.
        json_prefix_cursor(self.text)
        initial_sha = text_sha256(self.text)
        if not self.parts:
            self.parts.append(
                {
                    "sequence": 0,
                    "kind": "initial_partial",
                    "response_sha256": initial_sha,
                    "response_chars": len(self.text),
                    "response_utf8_bytes": len(self.text.encode("utf-8")),
                    "previous_document_sha256": None,
                    "expected_overlap_sha256": None,
                    "expected_overlap_chars": 0,
                    "appended_sha256": initial_sha,
                    "appended_chars": len(self.text),
                    "document_sha256": initial_sha,
                    "document_chars": len(self.text),
                }
            )
        self._seen_response_hashes.add(initial_sha)

    @property
    def continuation_count(self) -> int:
        return max(0, len(self.parts) - 1)

    def cursor(self) -> dict[str, Any]:
        overlap = self.text[-min(self.overlap_chars, len(self.text)) :]
        return {
            "version": STRUCTURED_CONTINUATION_VERSION,
            "document_id": self.document_id,
            "next_sequence": len(self.parts),
            "document_sha256": text_sha256(self.text),
            "document_chars": len(self.text),
            "expected_overlap": overlap,
            "expected_overlap_sha256": text_sha256(overlap),
            "expected_overlap_chars": len(overlap),
            "json_prefix": json_prefix_cursor(self.text),
        }

    def append(self, following: str, *, sequence: int) -> dict[str, Any]:
        cursor = self.cursor()
        if sequence != cursor["next_sequence"]:
            raise ValueError(
                "Continuation sequence mismatch: "
                f"expected {cursor['next_sequence']}, got {sequence}"
            )
        overlap = str(cursor["expected_overlap"])
        response_sha = text_sha256(following)
        if response_sha in self._seen_response_hashes:
            raise ValueError("Continuation repeated an already seen response")
        if not following.startswith(overlap):
            raise ValueError("Continuation has no exact literal boundary overlap")
        appended = following[len(overlap) :]
        if not appended:
            raise ValueError("Continuation made no forward progress")

        previous_sha = str(cursor["document_sha256"])
        candidate = self.text + appended
        candidate_sha = text_sha256(candidate)
        if candidate_sha == previous_sha:
            raise ValueError("Continuation did not change the document digest")
        # Reject impossible structural prefixes immediately.  Complete JSON is
        # parsed and schema-validated by the OpenRouter harness after a final
        # stop signal; this lexical check never turns a prefix into valid JSON.
        json_prefix_cursor(candidate)
        part = {
            "sequence": sequence,
            "kind": "literal_continuation",
            "response_sha256": response_sha,
            "response_chars": len(following),
            "response_utf8_bytes": len(following.encode("utf-8")),
            "previous_document_sha256": previous_sha,
            "expected_overlap_sha256": cursor["expected_overlap_sha256"],
            "expected_overlap_chars": cursor["expected_overlap_chars"],
            "appended_sha256": text_sha256(appended),
            "appended_chars": len(appended),
            "document_sha256": candidate_sha,
            "document_chars": len(candidate),
        }
        self.text = candidate
        self.parts.append(part)
        self._seen_response_hashes.add(response_sha)
        return dict(part)

    def manifest(self, *, complete: bool) -> dict[str, Any]:
        return {
            "version": STRUCTURED_CONTINUATION_VERSION,
            "mode": ResponseMode.CONTINUABLE_DOCUMENT.value,
            "document_id": self.document_id,
            "complete": bool(complete),
            "continuation_count": self.continuation_count,
            "part_count": len(self.parts),
            "document_sha256": text_sha256(self.text),
            "document_chars": len(self.text),
            "document_utf8_bytes": len(self.text.encode("utf-8")),
            "parts": [dict(part) for part in self.parts],
            "json_prefix": json_prefix_cursor(self.text),
        }


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _boundary(text: str, start: int, target_end: int) -> int:
    """Choose a readable boundary without dropping or duplicating a byte."""

    if target_end >= len(text):
        return len(text)
    minimum = start + max(1, (target_end - start) // 2)
    for marker in ("\n\n", "\n", ". ", "; ", ", ", " "):
        position = text.rfind(marker, minimum, target_end)
        if position >= minimum:
            return position + len(marker)
    return target_end


def _context_start(text: str, core_start: int, overlap_chars: int) -> int:
    """Expand left to a nearby line boundary without an unbounded paragraph."""

    if core_start <= 0 or overlap_chars <= 0:
        return core_start
    raw = max(0, core_start - overlap_chars)
    # Avoid beginning halfway through a Markdown heading/list line.  Looking
    # back at most 256 extra characters keeps the overlap a processing window,
    # never a corpus limit or an unbounded paragraph expansion.
    newline = text.rfind("\n", max(0, raw - 256), raw)
    return newline + 1 if newline >= 0 else raw


def _context_end(text: str, core_end: int, overlap_chars: int) -> int:
    """Expand right through a nearby line boundary within a fixed window."""

    if core_end >= len(text) or overlap_chars <= 0:
        return core_end
    raw = min(len(text), core_end + overlap_chars)
    newline = text.find("\n", raw, min(len(text), raw + 256))
    return newline + 1 if newline >= 0 else raw


def split_lossless_text(
    text: str,
    *,
    document_id: str,
    target_chars: int,
    context_overlap_chars: int = DEFAULT_CONTEXT_OVERLAP_CHARS,
) -> tuple[list[TextUnit], TextManifest]:
    """Partition text into exact cores plus semantic overlap context.

    ``TextUnit.text`` is the non-overlapping core and is the only field used
    for reconstruction. ``context_text`` is what an LLM should read: it adds a
    bounded prefix/suffix so an entity name, Markdown owner heading or
    owner-to-service relationship cannot disappear merely because it straddles
    a mechanical window boundary.
    """

    if target_chars < 256:
        raise ValueError("target_chars must be at least 256")
    if (
        isinstance(context_overlap_chars, bool)
        or not isinstance(context_overlap_chars, int)
        or context_overlap_chars < 0
    ):
        raise ValueError("context_overlap_chars must be a non-negative integer")
    spans: list[tuple[int, int]] = []
    if not text:
        # An empty source is still one explicit, attestable unit.  Returning a
        # zero-unit manifest would make it impossible for reducers to
        # distinguish "the source was empty" from "the source disappeared".
        spans.append((0, 0))
    start = 0
    while start < len(text):
        end = _boundary(text, start, min(len(text), start + target_chars))
        if end <= start:  # defensive: the loop must always make progress
            end = min(len(text), start + target_chars)
        spans.append((start, end))
        start = end

    units: list[TextUnit] = []
    for index, (start, end) in enumerate(spans):
        value = text[start:end]
        context_start = _context_start(text, start, context_overlap_chars)
        context_end = _context_end(text, end, context_overlap_chars)
        context = text[context_start:context_end]
        units.append(
            TextUnit(
                unit_id=f"{document_id}:{index:06d}",
                index=index,
                start_char=start,
                end_char=end,
                text=value,
                sha256=text_sha256(value),
                utf8_bytes=len(value.encode("utf-8")),
                context_start_char=context_start,
                context_end_char=context_end,
                context_text=context,
                context_sha256=text_sha256(context),
                context_utf8_bytes=len(context.encode("utf-8")),
                core_start_in_context=start - context_start,
                core_end_in_context=end - context_start,
            )
        )

    reconstructed = "".join(unit.text for unit in units)
    if reconstructed != text or text_sha256(reconstructed) != text_sha256(text):
        raise RuntimeError("Lossless text partition failed reconstruction")
    manifest = TextManifest(
        version=LOSSLESS_PARTITION_VERSION,
        document_id=document_id,
        source_sha256=text_sha256(text),
        source_chars=len(text),
        source_utf8_bytes=len(text.encode("utf-8")),
        unit_count=len(units),
        units=tuple(unit.metadata() for unit in units),
    )
    return units, manifest


def verify_units(
    units: Iterable[TextUnit],
    manifest: TextManifest | dict[str, Any],
) -> str:
    """Rebuild a document and fail closed on a missing, reordered or changed unit."""

    ordered = list(units)
    expected = (
        manifest.as_dict()
        if isinstance(manifest, TextManifest)
        else dict(manifest)
    )
    if [unit.index for unit in ordered] != list(range(len(ordered))):
        raise ValueError("Partition contains reordered, missing or duplicate units")
    if len(ordered) != int(expected.get("unit_count") or 0):
        raise ValueError("Partition unit count does not match its manifest")
    expected_units = expected.get("units")
    if not isinstance(expected_units, (list, tuple)):
        raise ValueError("Partition manifest has no unit metadata")
    if len(expected_units) != len(ordered):
        raise ValueError("Partition unit metadata count mismatch")
    cursor = 0
    for unit, expected_unit in zip(ordered, expected_units, strict=True):
        if not isinstance(expected_unit, dict):
            raise ValueError("Partition manifest unit metadata is invalid")
        if unit.start_char != cursor or unit.end_char < unit.start_char:
            raise ValueError("Partition character offsets are not contiguous")
        if text_sha256(unit.text) != unit.sha256:
            raise ValueError("Partition unit digest mismatch")
        for key, actual in (
            ("unit_id", unit.unit_id),
            ("index", unit.index),
            ("start_char", unit.start_char),
            ("end_char", unit.end_char),
            ("sha256", unit.sha256),
            ("utf8_bytes", unit.utf8_bytes),
            ("context_start_char", unit.context_start_char),
            ("context_end_char", unit.context_end_char),
            ("context_sha256", unit.context_sha256),
            ("context_utf8_bytes", unit.context_utf8_bytes),
            ("core_start_in_context", unit.core_start_in_context),
            ("core_end_in_context", unit.core_end_in_context),
        ):
            if expected_unit.get(key) != actual:
                raise ValueError(
                    f"Partition unit metadata mismatch for {unit.unit_id}: {key}"
                )
        if not (
            0 <= unit.context_start_char <= unit.start_char
            <= unit.end_char <= unit.context_end_char
        ):
            raise ValueError("Partition semantic context offsets are invalid")
        if unit.context_text[
            unit.core_start_in_context : unit.core_end_in_context
        ] != unit.text:
            raise ValueError("Partition core is not embedded in its context")
        if text_sha256(unit.context_text) != unit.context_sha256:
            raise ValueError("Partition context digest mismatch")
        cursor = unit.end_char
    text = "".join(unit.text for unit in ordered)
    if len(text) != int(expected.get("source_chars") or 0):
        raise ValueError("Reconstructed character count mismatch")
    if text_sha256(text) != str(expected.get("source_sha256") or ""):
        raise ValueError("Reconstructed document digest mismatch")
    return text


def partition_text_records(
    records: list[dict[str, Any]],
    *,
    text_key: str,
    id_key: str,
    target_chars: int,
    context_overlap_chars: int = DEFAULT_CONTEXT_OVERLAP_CHARS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Expand records into lossless code-owned units.

    Metadata is prefixed with ``_lr_`` so it can be excluded from a strict LLM
    schema while remaining available to the deterministic reducer.
    """

    expanded: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for record in records:
        document_id = str(record.get(id_key) or "")
        if not document_id:
            raise ValueError(f"Missing record id: {id_key}")
        source = str(record.get(text_key) or "")
        units, manifest = split_lossless_text(
            source,
            document_id=document_id,
            target_chars=target_chars,
            context_overlap_chars=context_overlap_chars,
        )
        manifests.append(manifest.as_dict())
        for unit in units:
            expanded.append(
                {
                    **record,
                    # The model reads bounded semantic context.  The exact,
                    # non-overlapping core remains separately attested and is
                    # the sole source of reconstruction/coverage accounting.
                    text_key: unit.context_text,
                    "_lr_core_text": unit.text,
                    "_lr_unit_id": unit.unit_id,
                    "_lr_unit_index": unit.index,
                    "_lr_unit_count": manifest.unit_count,
                    "_lr_unit_sha256": unit.sha256,
                    "_lr_source_sha256": manifest.source_sha256,
                    "_lr_source_chars": manifest.source_chars,
                    "_lr_start_char": unit.start_char,
                    "_lr_end_char": unit.end_char,
                    "_lr_context_start_char": unit.context_start_char,
                    "_lr_context_end_char": unit.context_end_char,
                    "_lr_context_sha256": unit.context_sha256,
                    "_lr_core_start_in_context": unit.core_start_in_context,
                    "_lr_core_end_in_context": unit.core_end_in_context,
                }
            )
    return expanded, manifests


def exact_boundary_join(previous: str, following: str, *, minimum: int = 32) -> tuple[str, int]:
    """Join narrative continuation only when a literal suffix/prefix overlaps."""

    maximum = min(len(previous), len(following))
    for size in range(maximum, minimum - 1, -1):
        if previous[-size:] == following[:size]:
            return previous + following[size:], size
    raise ValueError("Continuation has no exact boundary overlap")
