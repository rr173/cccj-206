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
    p.add_argument("--catalog-db", required=True,
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


if __name__ == "__main__":
    raise SystemExit(main())
