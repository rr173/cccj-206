"""Generate the small, committable demo dataset and its manifests.

Everything here is tiny (4x3 grids) so the outputs are safe to commit as small
samples; the generation command itself is the authoritative fixture source.
Large archives for memory testing are created separately in temp directories
by the test suite and are never committed.

Layout written under ``--out`` (default ./demo_data):

    <out>/inputs/<collection_id>/<file_id>.grib2
    <out>/manifests/<collection_id>.json
    <out>/request.json

Collections model the main drill:
    * c00            complete 00Z cycle (four cells)
    * c06_partial    06Z missing perturbed 3-6h
    * c06_complete   06Z all four cells
    * c06_conflict   06Z complete plus a byte-different copy of one cell
    * c06_nearmiss   06Z partial with 0-6h / other-grid / other-param decoys
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np  # noqa: E402

from fixturegen import FixtureSpec, encode_message, build_manifest  # noqa: E402
from demo_common import (  # noqa: E402
    CONTROL_SLOTS, PERTURBED_SLOTS, stat_spec, grid_digest_of,
)


def _write_file(root: Path, file_id: str, messages: list[bytes]) -> dict:
    rel = f"{file_id}.grib2"
    blob = b"".join(messages)
    (root / rel).write_bytes(blob)
    return {
        "file_id": file_id,
        "relative_path": rel,
        "size_bytes": len(blob),
        "sha256": hashlib.sha256(blob).hexdigest(),
    }


def _four_cell_specs(hour: int, *, skip_perturbed_36: bool = False):
    specs = []
    for member, slots in (("control", CONTROL_SLOTS),
                          ("perturbed", PERTURBED_SLOTS)):
        for slot_idx, (start, end) in enumerate(slots):
            if skip_perturbed_36 and member == "perturbed" and slot_idx == 1:
                continue
            specs.append(stat_spec(hour, member, start, end,
                                   data_seed=10 * hour + slot_idx
                                   + (1 if member == "perturbed" else 0)))
    return specs


def build(out: Path) -> None:
    inputs = out / "inputs"
    man_dir = out / "manifests"
    inputs.mkdir(parents=True, exist_ok=True)
    man_dir.mkdir(parents=True, exist_ok=True)

    grid_dig, _ = grid_digest_of()

    collections: dict[str, dict[str, list[FixtureSpec]]] = {
        "c00": {"all": _four_cell_specs(0)},
        "c06_partial": {"all": _four_cell_specs(6, skip_perturbed_36=True)},
        "c06_complete": {"all": _four_cell_specs(6)},
    }
    # Conflict collection: complete set plus a byte-different control 0-3h.
    conflict_files = {"all": _four_cell_specs(6)}
    conflict_files["dup_ctrl_03"] = [
        stat_spec(6, "control", 0, 3, data_seed=999)]
    collections["c06_conflict"] = conflict_files

    # Near-miss collection: partial 06 plus decoys that must not fill the gap.
    nearmiss = {"all": _four_cell_specs(6, skip_perturbed_36=True)}
    nearmiss["decoy_0_6h"] = [stat_spec(6, "perturbed", 0, 6, data_seed=7)]
    nearmiss["decoy_other_grid"] = [
        stat_spec(6, "perturbed", 3, 6, data_seed=8,
                  grid_overrides=dict(first_lat=55.0))]
    nearmiss["decoy_other_param"] = [
        stat_spec(6, "perturbed", 3, 6, data_seed=9, parameter_number=99)]
    collections["c06_nearmiss"] = nearmiss

    for cid, files in collections.items():
        root = inputs / cid
        root.mkdir(parents=True, exist_ok=True)
        entries = []
        for fid, specs in files.items():
            entries.append(_write_file(
                root, fid, [encode_message(s) for s in specs]))
        manifest = build_manifest(cid, root, entries,
                                  created_at="2026-01-01T00:00:00Z")
        (man_dir / f"{cid}.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")

    request = {
        "request_id": "drill",
        "origin_from": "2026-01-01T00:00:00Z",
        "origin_to": "2026-01-01T06:00:00Z",
        "not_after": "2026-01-01T12:00:00Z",
        "expected_grid_digest": grid_dig,
        "variables": [{
            "discipline": 0,
            "parameterCategory": 0,
            "parameterNumber": 0,
            "masterTablesVersionNumber": 29,
            "localTablesVersionNumber": 1,
            "centre": 7,
            "subCentre": 0,
        }],
        "levels": [{
            "first": {"type": 100, "scale_factor": 0,
                      "scaled_value": 85000},
            "second": None,
        }],
        "members": [
            {"kind": "control", "typeOfEnsembleForecast": 3,
             "perturbationNumber": 0},
            {"kind": "perturbed", "typeOfEnsembleForecast": 1,
             "perturbationNumber": 1},
        ],
        "slots": [
            {"start_seconds": 0, "end_seconds": 10800, "statistic": 1},
            {"start_seconds": 10800, "end_seconds": 21600, "statistic": 1},
        ],
    }
    (out / "request.json").write_text(
        json.dumps(request, indent=2), encoding="utf-8")
    print(f"wrote demo data to {out}")
    print(f"expected_grid_digest={grid_dig}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "demo_data"))
    args = ap.parse_args()
    build(Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
