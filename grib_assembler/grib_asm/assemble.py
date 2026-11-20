"""Assemble a hand-off package by copying encoded messages byte-for-byte.

The output directory is staged under a temporary name and only renamed to its
final, discoverable name once every record is written and validated, so a
failed or partial run can never be read as a complete result.  Messages are
concatenated in the exact request order (variable, level, member, slot) and
copied with a fixed buffer; packing is never changed and nothing is
re-encoded.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from .catalog import Catalog
from .errors import BlockedError
from .readiness import build_readiness, iter_request_keys
from .request import Manifest, Request
from .timeutil import iso
from datetime import datetime, timezone

COPY_BUFFER = 1024 * 1024
COMPLETE_MARKER = "PACKAGE_COMPLETE.json"
IN_PROGRESS_SUFFIX = ".in-progress"


def _now() -> str:
    return iso(datetime.now(timezone.utc))


def _stream_copy(src: Path, offset: int, length: int, out) -> str:
    """Copy ``length`` bytes from ``src[offset:]`` into ``out``; return hash."""

    h = hashlib.sha256()
    remaining = length
    with open(src, "rb") as fh:
        fh.seek(offset)
        while remaining:
            chunk = fh.read(min(COPY_BUFFER, remaining))
            if not chunk:
                raise IOError(f"unexpected EOF copying {src} at {offset}")
            h.update(chunk)
            out.write(chunk)
            remaining -= len(chunk)
    return h.hexdigest()


@dataclass
class PackageResult:
    package_dir: Path
    selected_reference_time: str
    record_count: int
    blocked: bool
    readiness: dict


def _resolve_source_path(input_roots: dict[str, Path], collection_id: str,
                         relative_path: str) -> Path:
    root = input_roots[collection_id]
    return root / relative_path


def assemble(catalog: Catalog, req: Request,
             input_roots: dict[str, Path], out_dir: str | Path,
             manifests: dict[str, Manifest],
             readiness_report=None) -> PackageResult:
    """Build a complete package or raise :class:`BlockedError`."""

    out_dir = Path(out_dir)
    if readiness_report is None:
        readiness_report = build_readiness(catalog, req)
    if readiness_report.blocked:
        raise BlockedError(readiness_report.to_dict())

    ref = readiness_report.selected_reference_time
    final_dir = out_dir / f"{req.request_id}__{ref.replace(':', '')}"
    stage_dir = final_dir.with_name(final_dir.name + IN_PROGRESS_SUFFIX)
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)

    records = []
    data_path = stage_dir / "messages.grib2"
    running_offset = 0
    try:
        with open(data_path, "wb") as out:
            for idx, key in iter_request_keys(req, ref):
                digest = key.digest()
                cell = catalog.cell(digest)
                if cell is None or cell["state"] != "filled":
                    # Defensive: readiness already guaranteed every cell.
                    raise RuntimeError(f"cell {digest} not filled at assemble")
                # Choose one source deterministically (first ordered source).
                src = cell["sources"][0]
                msg_row = catalog.message_row(
                    src["collection_id"], src["file_id"], src["msg_index"])
                if msg_row is None:
                    raise RuntimeError(
                        f"missing message row for {src}")
                source_path = _resolve_source_path(
                    input_roots, src["collection_id"],
                    _relative_path(catalog, src))
                length = int(msg_row["length"])
                offset = int(msg_row["offset"])
                expected_sha = msg_row["message_sha256"]

                start = running_offset
                actual_sha = _stream_copy(source_path, offset, length, out)
                if actual_sha != expected_sha:
                    raise IOError(
                        f"copy digest mismatch cell {idx}: "
                        f"{actual_sha} != {expected_sha}")
                running_offset += length
                records.append({
                    "cell_index": idx,
                    "output_offset": start,
                    "output_length": length,
                    "output_sha256": actual_sha,
                    "business_key_digest": digest,
                    "business_key": key.to_dict(),
                    "selected_reference_time": ref,
                    "source": {
                        "collection_id": src["collection_id"],
                        "file_id": src["file_id"],
                        "message_index": src["msg_index"],
                        "input_offset": offset,
                        "input_length": length,
                        "message_sha256": expected_sha,
                        "relative_path": _relative_path(catalog, src),
                    },
                    "all_retransmissions": cell["sources"],
                    "header_snapshot": json.loads(
                        msg_row["header_snapshot"]) if msg_row["header_snapshot"]
                    else {},
                })

        manifest_doc = {
            "package_format": "grib-assembler-handoff/v1",
            "request_id": req.request_id,
            "created_at": _now(),
            "selected_reference_time": ref,
            "not_after": iso(req.not_after),
            "record_order": [
                "variable", "level", "member", "slot"],
            "data_file": "messages.grib2",
            "record_count": len(records),
            "total_bytes": running_offset,
            "records": records,
        }
        (stage_dir / "package_manifest.json").write_text(
            json.dumps(manifest_doc, indent=2, sort_keys=False,
                       ensure_ascii=False),
            encoding="utf-8")

        # Copy source collection manifests so the package is self-contained.
        src_manifest_dir = stage_dir / "source_manifests"
        src_manifest_dir.mkdir(exist_ok=True)
        for cid, man in manifests.items():
            (src_manifest_dir / f"{cid}.json").write_text(
                json.dumps(man.raw, indent=2, ensure_ascii=False),
                encoding="utf-8")

        (stage_dir / "request.json").write_text(
            json.dumps(req.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8")
        (stage_dir / "readiness.json").write_text(
            json.dumps(readiness_report.to_dict(), indent=2,
                       ensure_ascii=False),
            encoding="utf-8")

        # Whole-package streaming hash over the final data file.
        whole = hashlib.sha256()
        with open(data_path, "rb") as fh:
            while True:
                chunk = fh.read(COPY_BUFFER)
                if not chunk:
                    break
                whole.update(chunk)
        marker = {
            "status": "complete",
            "finalized_at": _now(),
            "data_sha256": whole.hexdigest(),
            "record_count": len(records),
            "total_bytes": running_offset,
        }
        (stage_dir / COMPLETE_MARKER).write_text(
            json.dumps(marker, indent=2), encoding="utf-8")

        # Atomic publish: a reader only sees the directory once complete.
        os.replace(stage_dir, final_dir)
        return PackageResult(
            package_dir=final_dir,
            selected_reference_time=ref,
            record_count=len(records),
            blocked=False,
            readiness=readiness_report.to_dict(),
        )
    except Exception:
        if stage_dir.exists():
            shutil.rmtree(stage_dir)
        raise


def _relative_path(catalog: Catalog, src: dict) -> str:
    row = catalog._conn.execute(
        "SELECT relative_path FROM files WHERE collection_id=? AND file_id=?",
        (src["collection_id"], src["file_id"])).fetchone()
    if row is None:
        raise RuntimeError(f"no file row for {src}")
    return row["relative_path"]
