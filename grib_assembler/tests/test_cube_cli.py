"""End-to-end CLI smoke tests for the cube subcommands."""

from __future__ import annotations

import json

import pytest

from grib_asm.cube.cli import main as cube_main
from grib_asm.cube.importer import import_package

from cube_helpers import build_package, make_request, standard_grid_digest
from e2e_helpers import HAS_ECCODES

pytestmark = pytest.mark.skipif(
    not HAS_ECCODES, reason="ecCodes required to encode/decode fixtures")


def test_cli_import_inspect_query_recover(tmp_path, capsys):
    grid_dig = standard_grid_digest()
    request = make_request("cli", grid_dig)
    pkg = build_package(tmp_path, 0, request)
    store = tmp_path / "cube"

    rc = cube_main([
        "cube-import", "--store", str(store),
        "--package-dir", str(pkg), "--batch-size", "4"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "committed"
    assert out["origin"] == "2026-01-01T00:00:00Z"

    # Idempotent re-import through the CLI.
    rc = cube_main([
        "cube-import", "--store", str(store),
        "--package-dir", str(pkg)])
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "idempotent"

    # Inspection exposes committed coordinates and sources.
    rc = cube_main(["cube-inspect", "--store", str(store)])
    assert rc == 0
    inspect = json.loads(capsys.readouterr().out)
    assert inspect["origins"] == ["2026-01-01T00:00:00Z"]
    assert len(inspect["variables"]) == 1
    assert inspect["sources"]["2026-01-01T00:00:00Z"][
        "package_digest"]

    # Query with the package's request identity.
    rc = cube_main([
        "cube-query", "--store", str(store),
        "--request", str(pkg / "request.json"),
        "--variable-index", "0", "--include-values"])
    assert rc == 0
    query = json.loads(capsys.readouterr().out)
    assert query["present_shape"] == [1, 2, 2, 3]
    assert query["missing_count"] == 0
    assert len(query["values"]) == 1
    assert query["values"][0][0][0][0][0]

    # Recovery on a healthy store reports no work to do.
    rc = cube_main(["cube-recover", "--store", str(store)])
    assert rc == 0
    rec = json.loads(capsys.readouterr().out)
    assert rec["committed_origins"] == ["2026-01-01T00:00:00Z"]
    assert rec["orphaned_origins"] == []


def test_cli_unknown_variable_fails_without_side_effects(tmp_path, capsys):
    grid_dig = standard_grid_digest()
    request = make_request("cli2", grid_dig)
    pkg = build_package(tmp_path, 0, request)
    store = tmp_path / "cube"
    import_package(store, pkg)

    # Point the query at a variable index the request does not contain.
    with pytest.raises(SystemExit):
        cube_main([
            "cube-query", "--store", str(store),
            "--request", str(pkg / "request.json"),
            "--variable-index", "7"])
