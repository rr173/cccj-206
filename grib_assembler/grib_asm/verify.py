"""Independent hand-off package verifier.

Given only a hand-off directory (the original input tree is not needed), this
module:

* requires the completion marker, refusing to treat an in-progress directory
  as a result;
* streams ``messages.grib2`` in fixed buffers and recomputes the whole-file
  SHA-256 and every per-message SHA-256 from the recorded offsets;
* seeks to each record offset, hands exactly that encoded message to ecCodes
  and re-decodes the *header* fields (never data arrays), then checks them
  against the request and the stored canonical business key;
* confirms records appear in the required request order
  (variable, level, member, slot) and cover the full Cartesian product.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from .errors import GribAsmError
from .headers import MessageHeader, close_handle, open_handle_from_message_bytes
from .identity import BusinessKey
from .readiness import iter_request_keys
from .request import load_request
from .scanner import END_MARKER, HEADER_LEN, MAGIC
from .timeutil import iso

COPY_BUFFER = 1024 * 1024
COMPLETE_MARKER = "PACKAGE_COMPLETE.json"
IN_PROGRESS_SUFFIX = ".in-progress"


class VerificationError(GribAsmError):
    pass


@dataclass
class RecordCheck:
    index: int
    ok: bool
    problems: list[str] = field(default_factory=list)


@dataclass
class VerificationReport:
    ok: bool
    package_dir: str
    data_sha256: str
    record_count: int
    checks: list[RecordCheck]
    problems: list[str]

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "package_dir": self.package_dir,
            "data_sha256": self.data_sha256,
            "record_count": self.record_count,
            "problems": self.problems,
            "records": [
                {"index": c.index, "ok": c.ok, "problems": c.problems}
                for c in self.checks],
        }


def verify_package(package_dir: str | Path) -> VerificationReport:
    package_dir = Path(package_dir)
    problems: list[str] = []

    if package_dir.name.endswith(IN_PROGRESS_SUFFIX) or not package_dir.is_dir():
        raise VerificationError(
            f"{package_dir} is not a complete package directory")
    marker_path = package_dir / COMPLETE_MARKER
    if not marker_path.is_file():
        raise VerificationError(
            f"missing {COMPLETE_MARKER}; refusing to read incomplete package")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("status") != "complete":
        raise VerificationError("completion marker is not 'complete'")

    manifest = json.loads(
        (package_dir / "package_manifest.json").read_text(encoding="utf-8"))
    req = load_request(package_dir / "request.json")
    data_path = package_dir / manifest["data_file"]

    records = manifest["records"]
    checks: list[RecordCheck] = []

    # Whole-file streaming hash and per-offset boundary validation.
    whole = hashlib.sha256()
    with open(data_path, "rb") as fh:
        while True:
            chunk = fh.read(COPY_BUFFER)
            if not chunk:
                break
            whole.update(chunk)
    if whole.hexdigest() != marker["data_sha256"]:
        problems.append(
            f"whole data digest {whole.hexdigest()} != marker "
            f"{marker['data_sha256']}")

    expected_keys = [key for _idx, key in iter_request_keys(
        req, manifest["selected_reference_time"])]
    if len(expected_keys) != len(records):
        problems.append(
            f"record count {len(records)} != request cells "
            f"{len(expected_keys)}")

    total = 0
    for i, rec in enumerate(records):
        rc = RecordCheck(index=i, ok=True)
        off, length = rec["output_offset"], rec["output_length"]
        # Bounds and adjacency.
        if off != total:
            rc.problems.append(f"offset gap: {off} != expected {total}")
        total = off + length

        with open(data_path, "rb") as fh:
            fh.seek(off)
            raw = fh.read(length)
        if len(raw) != length:
            rc.problems.append("short read at record offset")
            checks.append(_fail(rc))
            continue
        if not raw.startswith(MAGIC) or not raw.endswith(END_MARKER):
            rc.problems.append("record lacks GRIB/7777 framing")
        mhash = hashlib.sha256(raw).hexdigest()
        if mhash != rec["output_sha256"]:
            rc.problems.append(
                f"record digest {mhash} != manifest {rec['output_sha256']}")
        if mhash != rec["source"]["message_sha256"]:
            rc.problems.append("output digest differs from source message digest")

        # Re-decode headers from the package bytes alone.
        handle = open_handle_from_message_bytes(raw)
        try:
            h = MessageHeader(handle)
            _check_header_against_key(h, raw, rec, expected_keys[i]
                                      if i < len(expected_keys) else None, rc)
        finally:
            close_handle(handle)

        if rc.problems:
            rc.ok = False
        checks.append(rc)

    if total != marker["total_bytes"]:
        problems.append(
            f"sum of records {total} != marker total_bytes "
            f"{marker['total_bytes']}")
    if total != data_path.stat().st_size:
        problems.append("data file size differs from summed record lengths")

    problems.extend(p for c in checks for p in
                    (f"record {c.index}: {x}" for x in c.problems))
    ok = not problems
    return VerificationReport(
        ok=ok,
        package_dir=str(package_dir),
        data_sha256=whole.hexdigest(),
        record_count=len(records),
        checks=checks,
        problems=problems,
    )


def _fail(rc: RecordCheck) -> RecordCheck:
    rc.ok = False
    return rc


def _check_header_against_key(h: MessageHeader, raw: bytes, rec: dict,
                              expected_key: BusinessKey | None,
                              rc: RecordCheck) -> None:
    from .gribbytes import parse_identification, parse_product
    if expected_key is None:
        rc.problems.append("no matching request cell for this index")
        return
    if rec["business_key_digest"] != expected_key.digest():
        rc.problems.append("record business key out of request order")

    # Numeric identity straight from the encoded bytes (build-independent).
    ident = parse_identification(raw)
    prod = parse_product(raw)
    want_var = expected_key.variable
    if (prod.discipline, prod.parameter_category, prod.parameter_number,
            ident.master_tables_version, ident.local_tables_version,
            ident.centre, ident.sub_centre) != (
            want_var.discipline, want_var.parameter_category,
            want_var.parameter_number, want_var.master_tables_version,
            want_var.local_tables_version, want_var.centre,
            want_var.sub_centre):
        rc.problems.append(
            f"variable identity mismatch: parsed "
            f"({prod.discipline},{prod.parameter_category},"
            f"{prod.parameter_number},{ident.master_tables_version},"
            f"{ident.local_tables_version},{ident.centre},"
            f"{ident.sub_centre})")

    # Reference time.
    from .timeutil import reference_datetime
    ref = reference_datetime(ident.data_date, ident.data_time)
    if iso(ref) != expected_key.reference_time:
        rc.problems.append("reference time mismatch")

    # Level first surface type and exact rational value.
    if prod.first_surface_type != expected_key.level.first.type:
        rc.problems.append("first fixed surface type mismatch")
    from .identity import scaled_value
    parsed_value = scaled_value(prod.first_scale_factor,
                                prod.first_scaled_value)
    if parsed_value != expected_key.level.first.value:
        rc.problems.append("first fixed surface value mismatch")
    # An undefined second surface must remain missing rather than become zero.
    if expected_key.level.second is None:
        if prod.second_surface_type != 255:
            rc.problems.append("expected missing second fixed surface")

    # Slot timing: PDT class and statistical process.
    slot = expected_key.slot
    pdt = prod.template_number
    if slot.is_instant:
        if pdt not in (0, 1):
            rc.problems.append(f"instant slot but PDT 4.{pdt}")
    else:
        if pdt not in (8, 11):
            rc.problems.append(f"statistical slot but PDT 4.{pdt}")
        # Cross-check the statistic via ecCodes scalar (with build alias).
        proc = None
        for candidate in ("statisticalProcess",
                          "typeOfStatisticalProcessing"):
            if h.has(candidate):
                try:
                    proc = h.int(candidate)
                    break
                except Exception:
                    continue
        if proc != slot.statistic:
            rc.problems.append(
                f"statisticalProcess {proc} != {slot.statistic}")
