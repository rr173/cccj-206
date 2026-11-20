"""Load and fully validate a hand-off package before any cube write.

Importing a package is allowed only when *all* of the following hold, and any
failure raises :class:`PackageValidationError` (or
:class:`UnsupportedCubeMessage`) without touching the published dataset:

1. the completion marker exists and says ``complete``; in-progress
   directories are refused;
2. the whole ``messages.grib2`` SHA-256 equals the marker digest;
3. every per-record offset/length framing and per-message SHA-256 is
   re-computed from the package bytes and matches the manifest;
4. the records form the exact request Cartesian product in the declared
   order (variable, level, member, slot), with contiguous offsets;
5. the original request identity stored in the package round-trips and every
   record's re-decoded headers match its canonical business key;
6. every record is a profile-supported message (the package verifier already
   guarantees integrity; here unsupported records make the cycle
   non-importable rather than being silently skipped).

The structural work is delegated to :mod:`grib_asm.verify` so that package
publication and cube import share one independent verifier; this module adds
the cube-specific identity and completeness rules on top.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..classify import classify_message
from ..headers import close_handle, open_handle_from_file
from ..identity import BusinessKey
from ..readiness import iter_request_keys
from ..request import load_request
from ..verify import (
    COMPLETE_MARKER, IN_PROGRESS_SUFFIX, verify_package,
)
from .errors import PackageValidationError, UnsupportedCubeMessage


@dataclass
class PackageRecord:
    index: int
    offset: int
    length: int
    message_sha256: str
    business_key: BusinessKey
    raw_header_snapshot: dict


@dataclass
class ValidatedPackage:
    package_dir: Path
    data_path: Path
    request_doc: dict
    request: object
    marker: dict
    package_digest: str          # marker whole-data SHA-256, identity of source
    records: list[PackageRecord]
    selected_reference_time: str

    def cell_count(self) -> int:
        return len(self.records)


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PackageValidationError("missing_package_file", {
            "path": str(path)}) from exc
    except json.JSONDecodeError as exc:
        raise PackageValidationError("malformed_package_json", {
            "path": str(path), "error": str(exc)}) from exc


def load_and_validate(package_dir: str | Path) -> ValidatedPackage:
    package_dir = Path(package_dir)
    if package_dir.name.endswith(IN_PROGRESS_SUFFIX) or not package_dir.is_dir():
        raise PackageValidationError("not_a_complete_package", {
            "package_dir": str(package_dir)})
    marker_path = package_dir / COMPLETE_MARKER
    if not marker_path.is_file():
        raise PackageValidationError("missing_completion_marker", {
            "package_dir": str(package_dir)})
    marker = _read_json(marker_path)
    if marker.get("status") != "complete":
        raise PackageValidationError("completion_marker_not_complete", {
            "status": marker.get("status")})

    # Shared independent verifier: framing, hashes, order, header re-decode.
    report = verify_package(package_dir)
    if not report.ok:
        raise PackageValidationError("package_verification_failed", {
            "problems": report.problems})

    manifest = _read_json(package_dir / "package_manifest.json")
    request = load_request(package_dir / "request.json")
    request_doc = _read_json(package_dir / "request.json")
    data_path = package_dir / manifest["data_file"]
    if not data_path.is_file():
        raise PackageValidationError("missing_data_file", {
            "data_file": str(data_path)})

    ref = manifest["selected_reference_time"]
    expected = [key for _idx, key in iter_request_keys(request, ref)]
    raw_records = manifest["records"]
    if len(raw_records) != len(expected):
        raise PackageValidationError("record_count_mismatch", {
            "records": len(raw_records), "request_cells": len(expected)})

    records: list[PackageRecord] = []
    running_offset = 0
    for i, (rec, want_key) in enumerate(zip(raw_records, expected)):
        off, length = int(rec["output_offset"]), int(rec["output_length"])
        if off != running_offset:
            raise PackageValidationError("non_contiguous_offsets", {
                "index": i, "offset": off,
                "expected": running_offset})
        running_offset = off + length
        if rec["business_key_digest"] != want_key.digest():
            raise PackageValidationError("message_order_mismatch", {
                "index": i,
                "record_digest": rec["business_key_digest"],
                "expected_digest": want_key.digest()})

        # Re-classify from the package bytes so the cube never imports a
        # message the active profile considers unsupported.  A *fresh* file
        # descriptor is opened per message: this ecCodes build cannot safely
        # reuse one native stream handle across consecutive messages (the
        # catalog uses exactly the same one-handle-per-message pattern).
        gds = _gds_bytes(data_path, off, length)
        with open(data_path, "rb") as fh:
            handle = open_handle_from_file(fh, off)
            try:
                norm = classify_message(
                    handle, None,
                    type("Span", (), {
                        "offset": off, "length": length,
                        "sha256": rec["output_sha256"],
                        "gds_bytes": gds,
                    })(),
                    file_id=package_dir.name,
                    collection_id="handoff",
                    fileobj=fh, msg_offset=off, msg_length=length)
            finally:
                close_handle(handle)
        if norm.status == "unsupported":
            raise UnsupportedCubeMessage(norm.reason or "unsupported", {
                "record_index": i,
                "business_key": want_key.to_dict(),
                "evidence": norm.header_snapshot.get(
                    "unsupported_evidence", {}),
            })
        if norm.business_key is None or \
                norm.business_key.digest() != want_key.digest():
            raise PackageValidationError("redecoded_identity_mismatch", {
                "index": i})
        records.append(PackageRecord(
                index=i, offset=off, length=length,
                message_sha256=rec["output_sha256"],
                business_key=want_key,
                raw_header_snapshot=norm.header_snapshot))

    if running_offset != marker.get("total_bytes"):
        raise PackageValidationError("total_bytes_mismatch", {
            "summed": running_offset,
            "marker": marker.get("total_bytes")})

    # Request identity: the stored request must match the manifest selection
    # and its grid digest must equal every record's grid identity.
    grid_digests = {r.business_key.grid_digest for r in records}
    if len(grid_digests) != 1:
        raise PackageValidationError("multiple_grid_definitions_in_package", {
            "grid_digests": sorted(grid_digests)})
    grid_digest = next(iter(grid_digests))
    if grid_digest != request.expected_grid_digest:
        raise PackageValidationError("request_grid_identity_mismatch", {
            "request_grid_digest": request.expected_grid_digest,
            "record_grid_digest": grid_digest})
    if request_doc.get("request_id") != manifest.get("request_id"):
        raise PackageValidationError("request_identity_mismatch", {
            "request_id_request": request_doc.get("request_id"),
            "request_id_manifest": manifest.get("request_id")})

    return ValidatedPackage(
        package_dir=package_dir, data_path=data_path,
        request_doc=request_doc, request=request, marker=marker,
        package_digest=marker["data_sha256"], records=records,
        selected_reference_time=ref)


def _gds_bytes(data_path: Path, offset: int, length: int) -> bytes:
    """Extract raw Section 3 bytes via a dedicated file descriptor.

    Only the small Section 3 is materialised; the data section is never read.
    A separate open keeps this Python-buffered read away from the ecCodes
    stream handle used on the main package file.
    """

    from ..scanner import _extract_gds
    with open(data_path, "rb") as gds_fh:
        return _extract_gds(gds_fh, offset, length)[0]
