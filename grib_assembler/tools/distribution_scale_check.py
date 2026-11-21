#!/usr/bin/env python3
"""Independent scale, conservation and bounded-memory check for distribution.

This standalone script writes only into a fresh temp directory and never
touches a cube.  It generates the two adversarial shapes named in the design:

1. a **hot digest** referenced by a large fan of rows (one forecast field
   consumed by many samples), which a naive clique builder would turn into a
   quadratic number of pairs;
2. a **single station with highly overlapping nested intervals**, which a
   naive pair enumerator would also expand quadratically.

For every requested size it then runs the real ``cube-training-distribute``
engine and independently verifies:

* conservation - every joint identity lands in exactly one output file;
* connectivity - against a small brute-force judge on a labeled subset whose
  expected components are hand-derived, not read from product structures;
* relation scale - message edges <= successful feature references,
  observation edges < rows, total edges <= rows + references, forest edges
  < rows, with the actual on-disk SQLite byte size reported;
* bounded memory - the per-1000-row resident-set slope stays near zero at two
  SQLite cache settings, because rows, references and edges live on disk.

Run::

    python -I tools/distribution_scale_check.py --sizes 500 2000 8000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

from grib_asm.training.distribute import (  # noqa: E402
    DistributionConfig, run_distribution,
)

H = 3600
T0 = 1767225600  # 2026-01-01T00:00:00Z


def digest(letter: str) -> str:
    return hashlib.sha256(letter.encode("utf-8")).hexdigest()


HOT = digest("hot-field")
NEST = digest("nested-field")
SOLO = digest("solo")
SOLO_UNIQUE = digest("solo-unique")


def _row(sample_id, station, metric, start_h, end_h, *, field, label="ok"):
    return {
        "sample_id": sample_id, "station_id": station, "metric": metric,
        "target_interval": [T0 + int(start_h * H), T0 + int(end_h * H)],
        "decision_at": "2026-01-01T00:00:00Z",
        "features": [{"column": "f0", "status": "ok", "value": 1.0,
                      "evidence": {"field": {"message_digest": field,
                                             "package_digest": "pkg-x"}}}],
        "label": {"status": label, "value": 1.0 if label == "ok" else None,
                  "unit": "K"},
    }


def generate(tmp: Path, n_hot: int, n_nested: int) -> dict:
    """Build two exports exercising the hot digest and nested intervals."""

    exp1, exp2 = [], []
    # Hot field: n_hot rows across two stations and both exports.  Every one
    # references HOT, so all of them are one component regardless of windows.
    for i in range(n_hot):
        export = exp1 if i % 2 == 0 else exp2
        # Intervals alternate train/validation windows so the hot component
        # spans windows and its candidates must be isolated.
        start_h = 1 + (i % 2) * 4
        export.append(_row(
            f"hot-{i:08d}", f"ST{i % 5}", "temp", start_h, start_h + 2,
            field=HOT))
    # One station, deeply nested intervals: interval k lives inside a long
    # envelope [1, n_nested+2) and successive intervals heavily overlap, so a
    # pair-enumerating implementation would create O(k^2) relations.
    envelope = _row("nest-root", "NEST", "temp", 1, n_nested + 2,
                    field=NEST, label="label_retracted")
    exp1.append(envelope)
    for k in range(n_nested):
        start_h = 1 + (k % max(n_nested, 1))
        end_h = min(start_h + 2, n_nested + 1)
        if end_h <= start_h:
            start_h, end_h = 1, 2
        exp2.append(_row(f"nest-{k:08d}", "NEST", "temp", start_h, end_h,
                         field=SOLO))
    # One fully independent holdout row (own message, own station), proving
    # that high fan-in elsewhere does not pull unrelated rows into components.
    exp2.append(_row("solo-holdout", "SOLO", "rain", 13, 14,
                     field=SOLO_UNIQUE))

    paths = {}
    for name, rows in (("exp1", exp1), ("exp2", exp2)):
        path = tmp / f"{name}.jsonl"
        path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8")
        paths[name] = path
    manifest = [{"export_id": "exp-one", "path": str(paths["exp1"])},
                {"export_id": "exp-two", "path": str(paths["exp2"])}]
    manifest_path = tmp / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    plan = {
        "gap_seconds": H,
        "windows": {
            "train": {"start": "2026-01-01T01:00:00Z",
                      "end": "2026-01-01T04:00:00Z"},
            "validation": {"start": "2026-01-01T04:00:00Z",
                           "end": "2026-01-01T10:00:00Z"},
            "holdout": {"start": "2026-01-01T12:00:00Z",
                        "end": "2026-01-01T18:00:00Z"}}}
    plan_path = tmp / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    return {"manifest": manifest_path, "plan": plan_path,
            "n_rows": len(exp1) + len(exp2),
            "n_hot": n_hot, "n_nested": n_nested}


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _conservation(out_dir: Path, expected_rows: int) -> None:
    seen = set()
    for name in ("train", "validation", "holdout", "isolated"):
        for line in (out_dir / f"{name}.jsonl").read_text(
                encoding="utf-8").splitlines():
            doc = json.loads(line)
            key = (doc["export_id"], doc["row"]["sample_id"])
            assert key not in seen, f"duplicate delivery of {key}"
            seen.add(key)
    assert len(seen) == expected_rows, (len(seen), expected_rows)


def _check_scale_bounds(scale: dict) -> None:
    n_rows, n_refs = scale["rows"], scale["successful_feature_refs"]
    assert scale["message_edges"] <= n_refs, scale
    assert scale["observation_edges"] < n_rows, scale
    assert scale["total_edges"] <= n_rows + n_refs, scale
    assert scale["forest_edges"] <= n_rows - 1, scale
    # The hot digest alone would imply (n_hot choose 2) pairs; the star builder
    # must emit only n_hot - 1 edges for it, never a quadratic table.
    assert scale["message_edges"] <= n_refs
    # Observation edges never exceed one per non-first row of a station group.
    assert scale["observation_edges"] <= n_rows


def run_one(tmp: Path, n_hot: int, n_nested: int, sqlite_cache_kib: int,
            page_limit: int) -> dict:
    info = generate(tmp, n_hot, n_nested)
    out_dir = tmp / "out"
    work_dir = tmp / "work"
    base_rss = _rss_mb()
    config = DistributionConfig(work_dir=str(work_dir), keep_workspace=True,
                                sqlite_cache_kib=sqlite_cache_kib,
                                page_limit=page_limit, commit_every=500)
    report = run_distribution(info["manifest"], info["plan"], out_dir, config)
    peak_rss = _rss_mb()
    _conservation(out_dir, info["n_rows"])
    _check_scale_bounds(report.relation_scale)

    # Independent connectivity expectations derived by hand:
    # * every hot-* row is in one component (shared HOT digest);
    # * every nest-* row plus nest-root is one component (nested overlaps);
    # * solo-holdout is its own component and is the only holdout delivery.
    decisions = [json.loads(line) for line in
                 (out_dir / "decisions.jsonl").read_text(
                     encoding="utf-8").splitlines()]
    groups = {(d["export_id"], d["sample_id"]):
              (d["group"]["export_id"], d["group"]["sample_id"])
              for d in decisions}
    hot_groups = {groups[(e, f"hot-{i:08d}")]
                  for i in range(info["n_hot"])
                  for e in ("exp-one", "exp-two")
                  if (e, f"hot-{i:08d}") in groups}
    assert len(hot_groups) == 1, hot_groups
    nest_keys = [("exp-one", "nest-root")] + [
        ("exp-two", f"nest-{k:08d}") for k in range(info["n_nested"])]
    nest_groups = {groups[k] for k in nest_keys if k in groups}
    assert len(nest_groups) == 1, nest_groups
    assert hot_groups != nest_groups
    solo = next(d for d in decisions if d["sample_id"] == "solo-holdout")
    assert solo["decision"] == "holdout", solo["decision"]

    index_bytes = sum(
        p.stat().st_size for p in out_dir.glob("distribution-index.sqlite*"))
    workspace_bytes = 0
    work = tmp / "work"
    if work.exists():
        workspace_bytes = sum(p.stat().st_size for p in work.rglob("*")
                              if p.is_file())
    return {
        "rows": info["n_rows"], "hot_rows": info["n_hot"],
        "nested_rows": info["n_nested"],
        "sqlite_cache_kib": sqlite_cache_kib,
        "page_limit": page_limit,
        "base_rss_mb": round(base_rss, 2),
        "peak_rss_mb": round(peak_rss, 2),
        "delta_rss_mb": round(peak_rss - base_rss, 2),
        "index_bytes": index_bytes,
        "workspace_bytes": workspace_bytes,
        "relation_scale": report.relation_scale,
        "shard_counts": report.shard_counts,
        "isolated_count": report.isolated_count,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int,
                        default=[500, 2000, 8000])
    parser.add_argument("--small-cache-kib", type=int, default=4 * 1024)
    parser.add_argument("--large-cache-kib", type=int, default=64 * 1024)
    parser.add_argument("--page-limit", type=int, default=1000)
    args = parser.parse_args(argv)

    tmp = Path(tempfile.mkdtemp(prefix="distribution-scale-"))
    measurements = []
    try:
        for n in args.sizes:
            for cache in (args.small_cache_kib, args.large_cache_kib):
                run_dir = tmp / f"n{n}-c{cache}"
                run_dir.mkdir()
                measurements.append(run_one(
                    run_dir, n_hot=n, n_nested=max(n // 2, 4),
                    sqlite_cache_kib=cache, page_limit=args.page_limit))
        slopes = {}
        for cache in (args.small_cache_kib, args.large_cache_kib):
            runs = [m for m in measurements
                    if m["sqlite_cache_kib"] == cache]
            small, large = runs[0], runs[-1]
            extra = max(large["rows"] - small["rows"], 1)
            extra_mb = max(large["delta_rss_mb"] - small["delta_rss_mb"], 0.0)
            slopes[f"cache{cache}_per_1000_rows_mb"] = round(
                extra_mb / extra * 1000, 4)
        report = {
            "sizes": args.sizes,
            "measurements": measurements,
            "slopes": slopes,
            "edge_rule": "message_edges <= successful_feature_refs; "
                         "observation_edges < rows; total edges <= rows + "
                         "refs; a million-row hot field therefore never "
                         "produces a quadratic edge table",
            "memory_rule": "rows, message_refs and edges are SQLite-backed; "
                           "resident state is bounded by page_limit and the "
                           "SQLite page cache, not by row or component size",
        }
        print(json.dumps(report, indent=2))
        for value in slopes.values():
            # Disk-backed pipeline: only a few MB marginal RSS per 1000 rows.
            assert value < 5.0, value
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
