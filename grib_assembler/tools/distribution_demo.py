#!/usr/bin/env python3
"""Generate and run the canonical connected-source distribution demo.

Everything is hand-authored and written into one fresh output directory; no
cube is opened and no forecast/station/observation matching is re-run.  The
inputs are JSONL rows shaped exactly like ``cube-training-export`` output (the
demo simply stands in for two prior export runs) plus an export manifest and a
delivery plan.

The hand-drawn picture (gap_seconds = 3600 s):

* ``A`` is a train-window candidate whose feature used message ``M1``;
* ``B`` has no usable label but its feature used the *same* ``M1`` (in a
  differently named column), so A and B share a source;
* ``C`` is a validation-window candidate on the same ``(station, metric)`` as
  B with an empty interval gap smaller than ``gap_seconds``, so B and C are
  observation-related; A and C have *no* direct relation;
* therefore A, B, C are one component spanning train and validation windows,
  and both candidates (A and C) must be isolated with an explainable chain;
* ``D`` shares C's station/metric but its empty gap is exactly
  ``gap_seconds``, which must not relate, so D is delivered on its own;
* ``E`` (same package, different message) and the cross-export ``M3`` pair
  demonstrate package-vs-message semantics; ``L`` is one long interval
  spanning several nested short intervals; ``H`` is an independent holdout.

Run::

    python -I tools/distribution_demo.py --out run/distribution-demo
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

from grib_asm.training.distribute import run_distribution  # noqa: E402
from grib_asm.training.distribute_explain import explain_row  # noqa: E402
from grib_asm.training.distribute_rows import joint_identity  # noqa: E402
from grib_asm.training.distribute_workspace import (  # noqa: E402
    DistributionWorkspace,
)

H = 3600
T0 = 1767225600  # 2026-01-01T00:00:00Z


def digest(letter: str) -> str:
    return hashlib.sha256(letter.encode("utf-8")).hexdigest()


M1 = digest("message-one")
M2 = digest("message-two")
M3 = digest("message-three")
M4 = digest("message-four")
M5 = digest("message-five")
M6 = digest("message-six")
M7 = digest("message-seven")
M8 = digest("message-eight")
M9 = digest("message-nine")


def row(sample_id, station, metric, start_h, end_h, *, field, package="pkg",
        column="f0", label="ok"):
    return {
        "sample_id": sample_id, "station_id": station, "metric": metric,
        "target_interval": [T0 + start_h * H, T0 + end_h * H],
        "decision_at": "2026-01-01T00:00:00Z",
        "features": [{"column": column, "status": "ok", "value": 1.25,
                      "evidence": {"field": {
                          "message_digest": field,
                          "package_digest": package}}}],
        "label": {"status": label,
                  "value": 2.0 if label == "ok" else None, "unit": "K"},
    }


def build_inputs(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    in_dir = out / "inputs"
    in_dir.mkdir(exist_ok=True)

    exp1 = [
        # Hand-drawn isolation chain A -M1-> B -obs-> C.
        row("A", "ST1", "temp", 1, 3, field=M1, column="temp_3_6h"),
        row("B", "ST1", "temp", 3, 4, field=M1, package="pkg-other",
            column="wind_3_6h", label="label_retracted"),
        row("C", "ST1", "temp", 4, 6, field=M2),
        # Exact-gap row: [7,8) vs C [4,6), empty gap == gap_seconds.  It also
        # uses a message nobody else consumes, so only the interval rule could
        # connect it - and by construction it does not.
        row("D", "ST1", "temp", 7, 8, field=M9),
        # Same package as C's M2 but a different message: no relation from
        # package identity alone (different station makes this airtight).
        row("E", "ST2", "temp", 13, 14, field=M6, package="pkg"),
        # One long interval L [1,9) with two nested short intervals on the
        # same station/metric; all three connect through L.
        row("L", "ST3", "temp", 1, 9, field=M4, label="label_retracted"),
        row("S1", "ST3", "temp", 2, 3, field=M7),
        row("S2", "ST3", "temp", 5, 6, field=M7),
        # Independent holdout candidate.
        row("H", "ST9", "rain", 13, 14, field=M8),
    ]
    exp2 = [
        # Same message M3 delivered in two different packages/exports.
        row("X", "ST7", "temp", 1, 2, field=M3, package="pkg-alpha"),
        row("Y", "ST8", "temp", 15, 16, field=M3, package="pkg-beta"),
        # Cross-window and outside-window rows.
        row("CW", "ST5", "temp", 3, 12, field=M4),
        row("OUT", "ST6", "temp", 20, 21, field=M4),
    ]
    p1 = in_dir / "export-2026-01.jsonl"
    p2 = in_dir / "export-2026-02.jsonl"
    p1.write_text("\n".join(json.dumps(r) for r in exp1) + "\n",
                  encoding="utf-8")
    p2.write_text("\n".join(json.dumps(r) for r in exp2) + "\n",
                  encoding="utf-8")
    manifest = [{"export_id": "export-2026-01", "path": str(p1)},
                {"export_id": "export-2026-02", "path": str(p2)}]
    manifest_path = in_dir / "export-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    plan = {
        "gap_seconds": H,
        "windows": {
            "train": {"start": "2026-01-01T01:00:00Z",
                      "end": "2026-01-01T04:00:00Z"},
            "validation": {"start": "2026-01-01T04:00:00Z",
                           "end": "2026-01-01T10:00:00Z"},
            "holdout": {"start": "2026-01-01T12:00:00Z",
                        "end": "2026-01-01T18:00:00Z"}}}
    plan_path = in_dir / "delivery-plan.json"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return {"manifest": manifest_path, "plan": plan_path}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True,
                        help="demo output directory")
    parser.add_argument("--keep-workspace", action="store_true")
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    paths = build_inputs(out)
    config_kwargs = {}
    if args.keep_workspace:
        config_kwargs["work_dir"] = str(out / "workspace")
        config_kwargs["keep_workspace"] = True
    from grib_asm.training.distribute import DistributionConfig
    report = run_distribution(
        paths["manifest"], paths["plan"], out / "delivery",
        DistributionConfig(**config_kwargs))
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))

    ws = DistributionWorkspace.open_readonly(
        out / "delivery" / "distribution-index.sqlite")
    try:
        explanation = explain_row(
            ws, joint_identity("export-2026-01", "A"))
    finally:
        ws.conn.close()
    print("\n# Explanation for isolated row A\n")
    print(json.dumps(explanation, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
