"""Command line interface for the hand-off package -> cube pipeline.

Subcommands:

* ``cube-import``  validate and import one hand-off package as one cycle;
* ``cube-recover`` reconcile uncommitted transaction work after a crash;
* ``cube-query``   select a variable slice with mask and provenance;
* ``cube-inspect`` print the committed coordinate registries.

The commands never open the ingest catalog and never touch source trees:
their only input is a completed, self-contained hand-off package directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..identity import Level, MemberId, VariableId
from ..request import load_request
from ..timeutil import TimeSlot
from ..training.errors import TrainingExportError
from .errors import CubeError
from .importer import ImportReport, import_package
from .query import CubeReader
from .recovery import recover
from .store import CubeConfig


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=False))


def _config(args) -> CubeConfig:
    return CubeConfig(
        batch_size=args.batch_size,
        coord_chunk=tuple(args.coord_chunk),
        compressor_level=args.compressor_level)


def cmd_import(args) -> int:
    report = import_package(
        args.store, args.package_dir, _config(args),
        recovery_mode=args.recovery_mode)
    _print(report.to_dict())
    return 0


def cmd_recover(args) -> int:
    report = recover(args.store, mode=args.recovery_mode)
    _print(report.to_dict())
    return 0


def cmd_inspect(args) -> int:
    reader = CubeReader(args.store)
    _print({
        "store": args.store,
        "meta": reader.meta,
        "origins": reader.origins(),
        "variables": reader.variables(),
        "levels": reader.levels(),
        "members": reader.members(),
        "slots": reader.slots(),
        "sources": {o: reader.source_of(o) for o in reader.origins()},
        "conflicts": reader.conflicts(),
    })
    return 0


def _identity_from_request(request_path: str | None, variable_index: int):
    if request_path is None:
        raise SystemExit("--request is required to identify the variable")
    req = load_request(request_path)
    if not 0 <= variable_index < len(req.variables):
        raise SystemExit(
            f"variable-index {variable_index} out of range "
            f"(0..{len(req.variables) - 1})")
    return req, variable_index


def cmd_query(args) -> int:
    req, var_idx = _identity_from_request(args.request, args.variable_index)
    var = req.variables[var_idx].as_variable_id()
    levels = None
    if args.level_index is not None:
        levels = [req.levels[i].as_level() for i in args.level_index]
    members = None
    if args.member_index is not None:
        members = [req.members[i].as_member_id() for i in args.member_index]
    slots = None
    if args.slot_index is not None:
        slots = [req.slots[i] for i in args.slot_index]
    reader = CubeReader(args.store)
    result = reader.select(
        var, levels=levels, members=members, slots=slots,
        origins=args.origin, validity_start_from=args.validity_start_from,
        validity_end_to=args.validity_end_to)
    payload = result.to_dict()
    if args.include_values:
        # Values/mask are emitted as nested lists on explicit request only;
        # normal inspection stays metadata-only.
        payload["values"] = result.values.tolist()
        payload["present"] = result.present.tolist()
        payload["message_digest"] = result.message_digest.tolist()
        payload["package_digest"] = result.package_digest.tolist()
    _print(payload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="grib-cube",
        description="Offline hand-off package to Zarr analysis cube.")
    sub = p.add_subparsers(dest="command", required=True)

    pi = sub.add_parser("cube-import",
                        help="validate and import one hand-off package")
    pi.add_argument("--store", required=True)
    pi.add_argument("--package-dir", required=True)
    pi.add_argument("--batch-size", type=int, default=16)
    pi.add_argument("--coord-chunk", nargs=4, type=int,
                    default=[1, 1, 1, 1], metavar=("O", "L", "M", "S"))
    pi.add_argument("--compressor-level", type=int, default=3)
    pi.add_argument("--recovery-mode", choices=("cleanup", "resume"),
                    default="cleanup")
    pi.set_defaults(func=cmd_import)

    pr = sub.add_parser("cube-recover",
                        help="reconcile uncommitted transaction work")
    pr.add_argument("--store", required=True)
    pr.add_argument("--recovery-mode", choices=("cleanup", "resume"),
                    default="cleanup")
    pr.set_defaults(func=cmd_recover)

    pn = sub.add_parser("cube-inspect",
                        help="print committed cube coordinates and sources")
    pn.add_argument("--store", required=True)
    pn.set_defaults(func=cmd_inspect)

    pq = sub.add_parser("cube-query", help="slice one variable from the cube")
    pq.add_argument("--store", required=True)
    pq.add_argument("--request", required=True,
                    help="request.json carrying the numeric variable identity")
    pq.add_argument("--variable-index", type=int, default=0)
    pq.add_argument("--level-index", type=int, action="append")
    pq.add_argument("--member-index", type=int, action="append")
    pq.add_argument("--slot-index", type=int, action="append")
    pq.add_argument("--origin", action="append",
                    help="reference time ...Z, repeatable")
    pq.add_argument("--validity-start-from")
    pq.add_argument("--validity-end-to")
    pq.add_argument("--include-values", action="store_true")
    pq.set_defaults(func=cmd_query)

    # Point-in-time training-table export (data preparation only).  The
    # subcommand parser lives in the training sub-package; only dispatch is
    # wired here.
    from ..training.export import ExportConfig

    pt = sub.add_parser(
        "cube-training-export",
        help="rebuild point-in-time features and cutoff labels into a "
             "training JSONL + summary")
    pt.add_argument("--store", required=True)
    pt.add_argument("--recipe", required=True)
    pt.add_argument("--samples", required=True)
    pt.add_argument("--publish-stream", required=True)
    pt.add_argument("--station-stream", required=True)
    pt.add_argument("--observation-stream", required=True)
    pt.add_argument("--output-jsonl", required=True)
    pt.add_argument("--output-summary", required=True)
    pt.add_argument("--work-dir")
    pt.add_argument("--keep-workspace", action="store_true")
    pt.add_argument("--sqlite-cache-kib", type=int, default=64 * 1024)
    pt.add_argument("--commit-every", type=int, default=1000)
    pt.add_argument("--point-batch", type=int, default=4096)

    def cmd_training_export(nargs) -> int:
        config = ExportConfig(
            work_dir=nargs.work_dir,
            keep_workspace=nargs.keep_workspace,
            sqlite_cache_kib=nargs.sqlite_cache_kib,
            commit_every=nargs.commit_every,
            point_batch=nargs.point_batch)
        from ..training.export import export_training_table
        report = export_training_table(
            store_path=nargs.store, recipe_path=nargs.recipe,
            samples_path=nargs.samples,
            publish_path=nargs.publish_stream,
            station_path=nargs.station_stream,
            observation_path=nargs.observation_stream,
            out_jsonl=nargs.output_jsonl, out_summary=nargs.output_summary,
            config=config)
        _print(report.to_dict())
        return 0

    pt.set_defaults(func=cmd_training_export)
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
