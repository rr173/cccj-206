"""Fault-injection tests for per-cycle commit atomicity.

The importer exposes named fail points (``CUBE_FAIL_POINT``) that call
``os._exit`` to simulate a hard process kill / power loss at three
boundaries:

* while a batch is being flushed into the staging group;
* between staging completion and registry swap;
* inside the commit, after array merge but before/after the marker swap.

After each kill a *fresh* process-independent reader must observe either
only previously committed cycles or the new cycle fully readable - never a
half-registered cycle.  Both recovery modes (cleanup and resume) are
exercised.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from grib_asm.cube.importer import import_package
from grib_asm.cube.query import CubeReader
from grib_asm.cube.recovery import recover
from grib_asm.cube.store import CUBE_DIR, CubeConfig

from cube_helpers import (
    build_package, expected_values, make_request, standard_grid_digest,
)
from e2e_helpers import HAS_ECCODES

pytestmark = pytest.mark.skipif(
    not HAS_ECCODES, reason="ecCodes required to encode/decode fixtures")

# Staging boundaries plus the three commit-internal boundaries.
FAIL_POINTS = (
    "during_batch_flush",
    "after_flush_before_prepare",
    "after_prepare_before_commit",
    "during_array_write",
    "during_registry_swap",
    "during_commit_marker",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_import_subprocess(store: Path, package: Path, fail_point: str | None,
                           root: Path):
    env = dict(os.environ)
    if fail_point:
        env["CUBE_FAIL_POINT"] = fail_point
    # Make the project importable explicitly.  A plain interpreter (not -I,
    # which would also ignore PYTHONPATH) keeps the subprocess independent
    # of the pytest process while running only this project's own code.
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + str(
        PROJECT_ROOT / "tools")
    code = r"""
import sys
from grib_asm.cube.importer import import_package
from grib_asm.cube.store import CubeConfig
store, package = sys.argv[1], sys.argv[2]
import_package(store, package, CubeConfig(batch_size=3))
"""
    proc = subprocess.run(
        [sys.executable, "-c", code, str(store), str(package)],
        cwd=root, env=env, capture_output=True, text=True)
    return proc


def _build_two_packages(tmp_path):
    grid_dig = standard_grid_digest()
    request = make_request("fault", grid_dig)
    pkg0 = build_package(tmp_path, 0, request)
    pkg1 = build_package(tmp_path, 1, request)
    return pkg0, pkg1


def test_hard_kill_leaves_no_half_committed_cycle(tmp_path):
    pkg0, pkg1 = _build_two_packages(tmp_path)
    store = tmp_path / "cube"

    # Cycle 0 commits cleanly in-process.
    assert import_package(store, pkg0).status == "committed"

    for fail_point in FAIL_POINTS:
        proc = _run_import_subprocess(store, pkg1, fail_point, tmp_path)
        assert proc.returncode == 42, proc.stderr

        # Immediately after the kill the new cycle must not be visible.
        reader = CubeReader(store)
        assert reader.origins() == ["2026-01-01T00:00:00Z"]
        from grib_asm.identity import VariableId
        result = reader.select(VariableId(0, 0, 0, 29, 1, 7, 0))
        # Only the first origin axis exists.
        assert result.values.shape[0] == 1
        assert result.present.sum() == result.present.size

        # Cleanup recovery removes staging and reconciles registries.
        report = recover(store, mode="cleanup")
        assert not list((store / CUBE_DIR / "txn").iterdir())
        assert reader.origins() == ["2026-01-01T00:00:00Z"]

        # The previously committed cycle is still fully readable.
        reader2 = CubeReader(store)
        r2 = reader2.select(VariableId(0, 0, 0, 29, 1, 7, 0))
        assert r2.present.sum() == r2.present.size

        # Re-import after recovery succeeds (idempotent staging was removed).
        again = import_package(store, pkg1, CubeConfig(batch_size=3))
        assert again.status == "committed"
        reader3 = CubeReader(store)
        assert reader3.origins() == [
            "2026-01-01T00:00:00Z", "2026-01-01T12:00:00Z"]
        # Reset for the next fail-point iteration.
        shutil.rmtree(store)
        import_package(store, pkg0)


def test_resume_mode_finishes_prepared_transaction(tmp_path):
    pkg0, pkg1 = _build_two_packages(tmp_path)
    store = tmp_path / "cube"
    import_package(store, pkg0)

    # Kill after prepare: the frozen snapshot allows resume to finish.
    proc = _run_import_subprocess(
        store, pkg1, "after_prepare_before_commit", tmp_path)
    assert proc.returncode == 42
    txn_dirs = list((store / CUBE_DIR / "txn").iterdir())
    assert txn_dirs and txn_dirs[0].is_dir()

    report = recover(store, mode="resume")
    assert len(report.resumed_transactions) == 1

    reader = CubeReader(store)
    assert reader.origins() == [
        "2026-01-01T00:00:00Z", "2026-01-01T12:00:00Z"]
    from grib_asm.identity import VariableId
    from cube_helpers import expected_values
    result = reader.select(VariableId(0, 0, 0, 29, 1, 7, 0))
    for li in range(2):
        for mi in range(2):
            for si in range(3):
                assert result.present[1, li, mi, si] == 1
                import numpy as np
                np.testing.assert_array_equal(
                    result.values[1, li, mi, si],
                    expected_values(1, li, mi, si))


def test_commit_marker_is_publish_boundary(tmp_path):
    pkg0, pkg1 = _build_two_packages(tmp_path)
    store = tmp_path / "cube"
    import_package(store, pkg0)

    # Kill inside the commit, after arrays but before the marker swap.
    proc = _run_import_subprocess(
        store, pkg1, "during_registry_swap", tmp_path)
    assert proc.returncode == 42

    # No marker means the reader must hide the origin even if registry/array
    # bytes for it reached disk.
    marker_dir = store / CUBE_DIR / "committed"
    markers = sorted(p.name for p in marker_dir.glob("*.json"))
    assert markers == ["20260101T000000Z.json"]

    recover(store)
    # After cleanup the orphan origin axis is truncated, not left dangling.
    reader = CubeReader(store)
    assert reader.origins() == ["2026-01-01T00:00:00Z"]
    from grib_asm.identity import VariableId
    result = reader.select(VariableId(0, 0, 0, 29, 1, 7, 0))
    assert result.present.shape[0] == 1


def test_first_cycle_kill_leaves_no_dataset_visible(tmp_path):
    pkg0, _pkg1 = _build_two_packages(tmp_path)
    store = tmp_path / "cube"

    # Killing the very first import means the dataset has no committed origin.
    for fail_point in ("during_batch_flush", "during_array_write",
                       "during_commit_marker"):
        proc = _run_import_subprocess(store, pkg0, fail_point, tmp_path)
        assert proc.returncode == 42, proc.stderr
        recover(store)
        # No commit marker: no origin may be readable.  When the kill happens
        # before the metadata step the store is not even a dataset yet.
        if not (store / CUBE_DIR / "CUBE_META.json").is_file():
            with pytest.raises(FileNotFoundError):
                CubeReader(store)
        else:
            assert CubeReader(store).origins() == []
        assert not list((store / CUBE_DIR / "txn").iterdir())
        # A later clean import still works from scratch.
        again = import_package(store, pkg0, CubeConfig(batch_size=3))
        assert again.status == "committed"
        assert CubeReader(store).origins() == [
            "2026-01-01T00:00:00Z"]
        shutil.rmtree(store)
