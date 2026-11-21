"""Command line entry for the training-data preparation commands.

Two independent offline stages live here:

* ``cube-training-export`` rebuilds point-in-time feature/label rows from a
  frozen cube and the four history streams;
* ``cube-training-distribute`` consumes already exported rows, builds the
  connected-source groups (shared GRIB2 messages and overlapping observation
  intervals) against a delivery plan and writes three shards plus one
  isolation file;
* ``cube-training-explain`` answers, from the delivered index alone, why one
  isolated joint identity could not be delivered and prints the re-checkable
  evidence chain to candidates in other windows.

Neither distribution command opens a cube, re-fetches points, re-runs the
history matching or trains a model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .distribute import DistributionConfig, run_distribution
from .distribute_errors import DistributionError
from .distribute_explain import explain_row
from .distribute_rows import joint_identity
from .distribute_workspace import DistributionWorkspace, WorkspaceConfig
from .errors import TrainingExportError

# The export command needs the cube stack (numpy/zarr); its imports live inside
# the handler so distribution/explain run on a standard-library-only host.


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=False))


def cmd_training_export(args) -> int:
    from .export import ExportConfig, export_training_table
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


def cmd_training_distribute(args) -> int:
    config = DistributionConfig(
        work_dir=args.work_dir,
        keep_workspace=args.keep_workspace,
        sqlite_cache_kib=args.sqlite_cache_kib,
        commit_every=args.commit_every,
        page_limit=args.page_limit)
    report = run_distribution(
        manifest_path=args.manifest,
        plan_path=args.plan,
        out_dir=args.out_dir,
        config=config)
    _print(report.to_dict())
    return 0


def cmd_training_explain(args) -> int:
    if args.joint_id is not None:
        asked = args.joint_id
    else:
        if args.export_id is None or args.sample_id is None:
            raise SystemExit(
                "provide either --joint-id or both --export-id and "
                "--sample-id")
        asked = joint_identity(args.export_id, args.sample_id)
    # Open the delivered index read-only: explanation must never mutate it and
    # must keep working once the original export files are gone.
    if not Path(args.index).is_file():
        raise FileNotFoundError(f"distribution index not found: {args.index}")
    ws_config = WorkspaceConfig(sqlite_cache_kib=args.sqlite_cache_kib)
    workspace = DistributionWorkspace.open_readonly(
        args.index, config=ws_config)
    try:
        result = explain_row(workspace, asked,
                             max_chain_nodes=args.max_chain_nodes)
    finally:
        workspace.conn.close()
    _print(result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cube-training",
        description="Offline training-data preparation: point-in-time "
                    "export, connected-source distribution and isolation "
                    "explanation.")
    sub = p.add_subparsers(dest="command", required=True)

    pe = sub.add_parser(
        "cube-training-export",
        prog="cube-training-export",
        description="Point-in-time feature/label training-table export from a "
                    "frozen analysis cube (data preparation only).")
    pe.add_argument("--store", required=True)
    pe.add_argument("--recipe", required=True)
    pe.add_argument("--samples", required=True)
    pe.add_argument("--publish-stream", required=True)
    pe.add_argument("--station-stream", required=True)
    pe.add_argument("--observation-stream", required=True)
    pe.add_argument("--output-jsonl", required=True)
    pe.add_argument("--output-summary", required=True)
    pe.add_argument("--work-dir")
    pe.add_argument("--keep-workspace", action="store_true")
    pe.add_argument("--sqlite-cache-kib", type=int, default=64 * 1024)
    pe.add_argument("--commit-every", type=int, default=1000)
    pe.add_argument("--point-batch", type=int, default=4096)
    pe.set_defaults(func=cmd_training_export)

    pd = sub.add_parser(
        "cube-training-distribute",
        prog="cube-training-distribute",
        description="Split already-exported rows into train/validation/"
                    "holdout shards plus one isolation file, honoring "
                    "transitive shared-source and overlapping-observation "
                    "groups. Never opens a cube or re-runs history matching.")
    pd.add_argument("--manifest", required=True,
                    help="JSON list of {export_id, path} export entries")
    pd.add_argument("--plan", required=True,
                    help="delivery plan JSON with gap_seconds and three "
                         "non-overlapping windows")
    pd.add_argument("--out-dir", required=True)
    pd.add_argument("--work-dir")
    pd.add_argument("--keep-workspace", action="store_true")
    pd.add_argument("--sqlite-cache-kib", type=int, default=16 * 1024)
    pd.add_argument("--commit-every", type=int, default=2000)
    pd.add_argument("--page-limit", type=int, default=2000)
    pd.set_defaults(func=cmd_training_distribute)

    px = sub.add_parser(
        "cube-training-explain",
        prog="cube-training-explain",
        description="Explain from the delivered index why one isolated row "
                    "could not be distributed, with re-checkable evidence "
                    "chains to candidates in other windows.")
    px.add_argument("--index", required=True,
                    help="distribution-index.sqlite produced by "
                         "cube-training-distribute")
    px.add_argument("--joint-id",
                    help="full joint identity (export_id + ASCII unit "
                         "separator + sample_id)")
    px.add_argument("--export-id")
    px.add_argument("--sample-id")
    px.add_argument("--sqlite-cache-kib", type=int, default=16 * 1024)
    px.add_argument("--max-chain-nodes", type=int, default=5000,
                    help="resident-node budget per evidence chain before the "
                         "logarithmic spanning-forest path is used")
    px.set_defaults(func=cmd_training_explain)
    return p


SUBCOMMANDS = (
    "cube-training-export",
    "cube-training-distribute",
    "cube-training-explain",
)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    if argv is None:
        argv = sys.argv[1:]
    # Backward compatibility: the historical entry point and module invocation
    # accepted export flags directly, with no leading subcommand.  Treat such a
    # call as ``cube-training-export`` so existing runners keep working while
    # the new subcommands remain explicitly selectable.
    if not argv or argv[0] not in SUBCOMMANDS:
        argv = ["cube-training-export", *argv]
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except TrainingExportError as exc:
        _print({"error": exc.reason, "evidence": exc.evidence})
        return 6
    except DistributionError as exc:
        _print({"error": exc.reason, "evidence": exc.evidence})
        return 7
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        # CubeError lives behind the optional cube stack; match structurally.
        if getattr(exc, "reason", None) is not None and \
                exc.__class__.__name__ == "CubeError":
            _print({"error": exc.reason, "evidence": getattr(exc,
                                                              "evidence", {})})
            return 6
        raise


if __name__ == "__main__":
    raise SystemExit(main())
