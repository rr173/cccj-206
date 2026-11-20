"""Command line entry for the point-in-time training export.

``cube-training-export`` reads one frozen cube, one recipe and four
order-independent JSONL streams and writes a directly consumable training
JSONL plus a count summary.  It performs no modelling, forecasting,
interpolation or unit conversion.
"""

from __future__ import annotations

import argparse
import json
import sys

from ..cube.errors import CubeError
from .errors import TrainingExportError
from .export import ExportConfig, export_training_table


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=False))


def cmd_training_export(args) -> int:
    config = ExportConfig(
        work_dir=args.work_dir,
        keep_workspace=args.keep_workspace,
        sqlite_cache_kib=args.sqlite_cache_kib,
        commit_every=args.commit_every,
        point_batch=args.point_batch)
    report = export_training_table(
        store_path=args.store,
        recipe_path=args.recipe,
        samples_path=args.samples,
        publish_path=args.publish_stream,
        station_path=args.station_stream,
        observation_path=args.observation_stream,
        out_jsonl=args.output_jsonl,
        out_summary=args.output_summary,
        config=config)
    _print(report.to_dict())
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cube-training-export",
        description="Point-in-time feature/label training-table export from a "
                    "frozen analysis cube (data preparation only).")
    p.add_argument("--store", required=True, help="frozen Zarr cube store")
    p.add_argument("--recipe", required=True, help="feature recipe JSON")
    p.add_argument("--samples", required=True, help="sample manifest JSONL")
    p.add_argument("--publish-stream", required=True,
                   help="forecast publish/withdraw JSONL (any order)")
    p.add_argument("--station-stream", required=True,
                   help="station map/unmap JSONL (any order)")
    p.add_argument("--observation-stream", required=True,
                   help="measurement set/retract JSONL (any order)")
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--output-summary", required=True)
    p.add_argument("--work-dir",
                   help="cleanable disk workspace (default: temp directory)")
    p.add_argument("--keep-workspace", action="store_true",
                   help="retain the SQLite workspace for inspection")
    p.add_argument("--sqlite-cache-kib", type=int, default=64 * 1024)
    p.add_argument("--commit-every", type=int, default=1000)
    p.add_argument("--point-batch", type=int, default=4096)
    p.set_defaults(func=cmd_training_export)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except TrainingExportError as exc:
        _print({"error": exc.reason, "evidence": exc.evidence})
        return 6
    except CubeError as exc:
        _print({"error": exc.reason, "evidence": exc.evidence})
        return 6
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
