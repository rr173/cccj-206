"""Command line entry points.

Subcommands are independently runnable and share only the on-disk catalog:

* ``ingest``      scan manifest files into the catalog (streaming, no values)
* ``readiness``   per-cycle readiness, missing/conflict reasons, selection
* ``assemble``    build a self-contained hand-off package, or report blocked
* ``explain-slot`` explain one cell independently
* ``verify-package`` independently re-seek, re-decode headers and re-hash
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .assemble import assemble
from .catalog import Catalog
from .errors import BlockedError
from .explain import explain_slot
from .readiness import build_readiness
from .request import load_manifest, load_request
from .verify import verify_package


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=False))


def _open_catalog(args) -> tuple[Catalog, dict]:
    if not args.catalog_db:
        raise SystemExit("--catalog-db is required for this subcommand")
    cat = Catalog(args.catalog_db)
    roots = {}
    if args.input_roots:
        for pair in args.input_roots:
            cid, root = pair.split("=", 1)
            roots[cid] = Path(root)
    return cat, roots


def cmd_ingest(args) -> int:
    cat, roots = _open_catalog(args)
    results = []
    for manifest_path in args.manifests:
        manifest = load_manifest(manifest_path)
        if manifest.collection_id not in roots:
            raise SystemExit(
                f"missing --input-roots {manifest.collection_id}=<path>")
        for entry in manifest.files:
            res = cat.ingest_file(
                manifest, entry, roots[manifest.collection_id],
                bufsize=args.bufsize)
            results.append(vars(res))
    cat.close()
    _print({"ingested": results})
    return 0


def cmd_readiness(args) -> int:
    cat, _ = _open_catalog(args)
    req = load_request(args.request)
    report = build_readiness(cat, req)
    cat.close()
    doc = report.to_dict()
    if args.output:
        Path(args.output).write_text(
            json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    _print(doc)
    return 0 if not doc["blocked"] else 3


def cmd_assemble(args) -> int:
    cat, roots = _open_catalog(args)
    req = load_request(args.request)
    manifests = {p: load_manifest(p) for p in args.manifests}
    by_id = {m.collection_id: m for m in manifests.values()}
    report = build_readiness(catalog=cat, req=req)
    if report.blocked:
        doc = report.to_dict()
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        block_path = Path(args.out_dir) / f"{req.request_id}.blocked.json"
        block_path.write_text(
            json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
        cat.close()
        _print({"blocked": True, "report": doc, "written": str(block_path)})
        return 4
    result = assemble(cat, req, roots, args.out_dir, by_id,
                      readiness_report=report)
    cat.close()
    _print({
        "blocked": False,
        "package_dir": str(result.package_dir),
        "selected_reference_time": result.selected_reference_time,
        "record_count": result.record_count,
    })
    return 0


def cmd_explain(args) -> int:
    cat, _ = _open_catalog(args)
    req = load_request(args.request)
    expl = explain_slot(cat, req, args.reference_time,
                        args.variable_index, args.level_index,
                        args.member_index, args.slot_index)
    cat.close()
    _print(expl.to_dict())
    return 0


def cmd_verify(args) -> int:
    report = verify_package(args.package_dir)
    _print(report.to_dict())
    return 0 if report.ok else 5


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="grib-asm",
        description="Offline GRIB2 collection assembler (headers/copy only).")
    # The catalog is only needed by the ingest/readiness/assemble family;
    # cube subcommands operate on completed hand-off packages alone.
    p.add_argument("--catalog-db",
                   help="SQLite catalog path (disk-backed state).")
    p.add_argument("--input-roots", action="append", default=[],
                   help="collection_id=/abs/input/root, repeatable.")
    p.add_argument("--bufsize", type=int, default=1024 * 1024)
    sub = p.add_subparsers(dest="command", required=True)

    pi = sub.add_parser("ingest")
    pi.add_argument("--manifests", nargs="+", required=True)
    pi.set_defaults(func=cmd_ingest)

    pr = sub.add_parser("readiness")
    pr.add_argument("--request", required=True)
    pr.add_argument("--output")
    pr.set_defaults(func=cmd_readiness)

    pa = sub.add_parser("assemble")
    pa.add_argument("--request", required=True)
    pa.add_argument("--manifests", nargs="+", required=True)
    pa.add_argument("--out-dir", required=True)
    pa.set_defaults(func=cmd_assemble)

    pe = sub.add_parser("explain-slot")
    pe.add_argument("--request", required=True)
    pe.add_argument("--reference-time", required=True)
    pe.add_argument("--variable-index", type=int, required=True)
    pe.add_argument("--level-index", type=int, required=True)
    pe.add_argument("--member-index", type=int, required=True)
    pe.add_argument("--slot-index", type=int, required=True)
    pe.set_defaults(func=cmd_explain)

    pv = sub.add_parser("verify-package")
    pv.add_argument("--package-dir", required=True)
    pv.set_defaults(func=cmd_verify)

    # Package -> analysis cube family.  These delegate to grib_asm.cube and
    # do not use the ingest catalog.
    from .cube.cli import (
        cmd_import as cube_cmd_import,
        cmd_inspect as cube_cmd_inspect,
        cmd_query as cube_cmd_query,
        cmd_recover as cube_cmd_recover,
    )

    pc = sub.add_parser("cube-import")
    pc.add_argument("--store", required=True)
    pc.add_argument("--package-dir", required=True)
    pc.add_argument("--batch-size", type=int, default=16)
    pc.add_argument("--coord-chunk", nargs=4, type=int,
                    default=[1, 1, 1, 1], metavar=("O", "L", "M", "S"))
    pc.add_argument("--compressor-level", type=int, default=3)
    pc.add_argument("--recovery-mode", choices=("cleanup", "resume"),
                    default="cleanup")
    pc.set_defaults(func=cube_cmd_import)

    pcr = sub.add_parser("cube-recover")
    pcr.add_argument("--store", required=True)
    pcr.add_argument("--recovery-mode", choices=("cleanup", "resume"),
                     default="cleanup")
    pcr.set_defaults(func=cube_cmd_recover)

    pci = sub.add_parser("cube-inspect")
    pci.add_argument("--store", required=True)
    pci.set_defaults(func=cube_cmd_inspect)

    pcq = sub.add_parser("cube-query")
    pcq.add_argument("--store", required=True)
    pcq.add_argument("--request", required=True)
    pcq.add_argument("--variable-index", type=int, default=0)
    pcq.add_argument("--level-index", type=int, action="append")
    pcq.add_argument("--member-index", type=int, action="append")
    pcq.add_argument("--slot-index", type=int, action="append")
    pcq.add_argument("--origin", action="append")
    pcq.add_argument("--validity-start-from")
    pcq.add_argument("--validity-end-to")
    pcq.add_argument("--include-values", action="store_true")
    pcq.set_defaults(func=cube_cmd_query)

    # Point-in-time training-table export.  The handler lives in the training
    # sub-package; the top-level CLI only mirrors its arguments and dispatches.
    from .training.export import ExportConfig

    pt = sub.add_parser(
        "cube-training-export",
        help="point-in-time feature/label training-table export")
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

    def _training_handler(nargs) -> int:
        from .training.export import export_training_table
        config = ExportConfig(
            work_dir=nargs.work_dir,
            keep_workspace=nargs.keep_workspace,
            sqlite_cache_kib=nargs.sqlite_cache_kib,
            commit_every=nargs.commit_every,
            point_batch=nargs.point_batch)
        report = export_training_table(
            store_path=nargs.store, recipe_path=nargs.recipe,
            samples_path=nargs.samples,
            publish_path=nargs.publish_stream,
            station_path=nargs.station_stream,
            observation_path=nargs.observation_stream,
            out_jsonl=nargs.output_jsonl,
            out_summary=nargs.output_summary, config=config)
        _print(report.to_dict())
        return 0

    pt.set_defaults(func=_training_handler)

    # Offline distribution of already-exported rows and isolation explanation.
    # Both commands delegate to the training sub-package and never open a cube.
    from .training.distribute import DistributionConfig
    from .training.distribute_workspace import (
        DistributionWorkspace, WorkspaceConfig,
    )
    from .training.distribute_explain import explain_row
    from .training.distribute_rows import joint_identity

    pdist = sub.add_parser(
        "cube-training-distribute",
        help="split exported rows into train/validation/holdout plus an "
             "isolation file using connected-source groups")
    pdist.add_argument("--manifest", required=True)
    pdist.add_argument("--plan", required=True)
    pdist.add_argument("--out-dir", required=True)
    pdist.add_argument("--work-dir")
    pdist.add_argument("--keep-workspace", action="store_true")
    pdist.add_argument("--sqlite-cache-kib", type=int, default=16 * 1024)
    pdist.add_argument("--commit-every", type=int, default=2000)
    pdist.add_argument("--page-limit", type=int, default=2000)

    def _distribute_handler(nargs) -> int:
        config = DistributionConfig(
            work_dir=nargs.work_dir,
            keep_workspace=nargs.keep_workspace,
            sqlite_cache_kib=nargs.sqlite_cache_kib,
            commit_every=nargs.commit_every,
            page_limit=nargs.page_limit)
        from .training.distribute import run_distribution
        report = run_distribution(
            manifest_path=nargs.manifest, plan_path=nargs.plan,
            out_dir=nargs.out_dir, config=config)
        _print(report.to_dict())
        return 0

    pdist.set_defaults(func=_distribute_handler)

    pexpl = sub.add_parser(
        "cube-training-explain",
        help="explain an isolated row from the delivered distribution index")
    pexpl.add_argument("--index", required=True)
    pexpl.add_argument("--joint-id")
    pexpl.add_argument("--export-id")
    pexpl.add_argument("--sample-id")
    pexpl.add_argument("--sqlite-cache-kib", type=int, default=16 * 1024)
    pexpl.add_argument("--max-chain-nodes", type=int, default=5000,
                       help="resident-node budget per evidence chain before "
                            "the logarithmic spanning-forest path is used")

    def _explain_handler(nargs) -> int:
        if nargs.joint_id is not None:
            asked = nargs.joint_id
        else:
            if nargs.export_id is None or nargs.sample_id is None:
                raise SystemExit(
                    "provide either --joint-id or both --export-id and "
                    "--sample-id")
            asked = joint_identity(nargs.export_id, nargs.sample_id)
        if not Path(nargs.index).is_file():
            raise FileNotFoundError(
                f"distribution index not found: {nargs.index}")
        ws = DistributionWorkspace.open_readonly(
            nargs.index,
            config=WorkspaceConfig(sqlite_cache_kib=nargs.sqlite_cache_kib))
        try:
            _print(explain_row(ws, asked,
                               max_chain_nodes=nargs.max_chain_nodes))
        finally:
            ws.conn.close()
        return 0

    pexpl.set_defaults(func=_explain_handler)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except BlockedError as exc:
        _print({"blocked": True, "report": exc.report})
        return 4
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        # Cube pipeline, training-export and distribution errors carry a
        # stable reason code.
        reason = getattr(exc, "reason", None)
        if reason is not None:
            _print({"error": reason, "evidence": getattr(exc, "evidence", {})})
            return 7 if reason.startswith(("plan_", "manifest_", "row_",
                                           "gap_", "duplicate_export",
                                           "successful_feature",
                                           "explain_")) else 6
        raise


if __name__ == "__main__":
    raise SystemExit(main())
