"""Crash recovery for uncommitted cube work.

A reader only ever trusts origins that have both registry entries *and* a
per-origin commit marker.  Recovery reconciles the on-disk state after an
interruption (process kill, disk full, power loss):

* ``committed`` journals are stale leftovers of a finished transaction and
  are simply removed;
* ``open``/``prepared`` journals without a commit marker describe work that
  never became visible.  In ``cleanup`` mode (default) their staging is
  deleted and any registry/array slab that reached disk ahead of the marker
  is truncated; in ``resume`` mode a *prepared* transaction is finished from
  its frozen registry snapshot instead of being discarded;
* already committed cycles are never rewritten or re-hashed.

Because origins are append-only, an orphaned origin (registry entry without
a marker) can only ever be the last index, so truncation cannot disturb
committed coordinates.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr

from .errors import CubeError
from .store import (
    COMMIT_DIR, CUBE_DIR, DIGEST_DTYPE, TXN_DIR, _atomic_write_json,
    _load_json, marker_name,
)


@dataclass
class RecoveryReport:
    removed_transactions: list[str]
    resumed_transactions: list[str]
    orphaned_origins: list[str]
    committed_origins: list[str]

    def to_dict(self) -> dict:
        return {
            "removed_transactions": self.removed_transactions,
            "resumed_transactions": self.resumed_transactions,
            "orphaned_origins": self.orphaned_origins,
            "committed_origins": self.committed_origins,
        }


def marker_exists(root: Path, origin: str) -> bool:
    return (root / CUBE_DIR / COMMIT_DIR
            / f"{marker_name(origin)}.json").is_file()


def _committed_origins(root: Path) -> list[str]:
    out = []
    commit_dir = root / CUBE_DIR / COMMIT_DIR
    if not commit_dir.is_dir():
        return out
    for p in sorted(commit_dir.glob("*.json")):
        out.append(json.loads(p.read_text(encoding="utf-8"))["origin"])
    return out


def _reopen_or_create(group, var_key_name: str, leaf: str, shape, *,
                      dtype, fill):
    """Open a published array, creating it from the snapshot shape if absent."""

    path = f"vars/{var_key_name}/{leaf}"
    existing = group.get(path)
    if existing is not None:
        if tuple(existing.shape) != tuple(shape):
            existing.resize(shape)
        return existing
    if dtype == "float64":
        chunks = shape[:-2] + (shape[-2], shape[-1])
    else:
        chunks = shape
    return group.create_dataset(path, shape=shape, chunks=chunks,
                                dtype=dtype, fill_value=fill)


def recover(store_path: str | Path, mode: str = "cleanup") -> RecoveryReport:
    """Reconcile transaction state on disk.

    ``mode`` is ``"cleanup"`` (discard uncommitted work) or ``"resume"``
    (finish prepared transactions from their frozen snapshots).
    """

    if mode not in ("cleanup", "resume"):
        raise CubeError("invalid_recovery_mode", {"mode": mode})
    root = Path(store_path)
    txn_root = root / CUBE_DIR / TXN_DIR
    report = RecoveryReport([], [], [], _committed_origins(root))

    if not root.exists():
        return report

    if txn_root.is_dir():
        for txn_dir in sorted(p for p in txn_root.iterdir() if p.is_dir()):
            journal_path = txn_dir / "journal.json"
            if not journal_path.is_file():
                # No journal at all: nothing was registered, staging is junk.
                shutil.rmtree(txn_dir, ignore_errors=True)
                report.removed_transactions.append(txn_dir.name)
                continue
            journal = json.loads(journal_path.read_text(encoding="utf-8"))
            origin = journal["origin"]
            if journal.get("state") == "committed" or \
                    marker_exists(root, origin):
                # The cycle is visible; the directory is post-commit litter.
                shutil.rmtree(txn_dir, ignore_errors=True)
                report.removed_transactions.append(txn_dir.name)
                continue
            if mode == "resume" and journal.get("state") == "prepared":
                _resume_prepared(root, txn_dir, journal, report)
            else:
                shutil.rmtree(txn_dir, ignore_errors=True)
                report.removed_transactions.append(txn_dir.name)

    _reconcile_registry(root, report)
    return report


def _resume_prepared(root: Path, txn_dir: Path, journal: dict,
                     report: RecoveryReport) -> None:
    """Finish a prepared transaction using its frozen registry snapshot."""

    snap_dir = txn_dir / "registry_snapshot"
    if not snap_dir.is_dir():
        shutil.rmtree(txn_dir, ignore_errors=True)
        report.removed_transactions.append(txn_dir.name)
        return
    origin = journal["origin"]
    if marker_exists(root, origin):
        shutil.rmtree(txn_dir, ignore_errors=True)
        report.removed_transactions.append(txn_dir.name)
        return

    snapshot = {p.name: json.loads(p.read_text(encoding="utf-8"))
                for p in snap_dir.glob("*.json")}
    registry_dir = root / CUBE_DIR / "registry"

    stage = zarr.open_group(str(txn_dir / "stage"), mode="r")
    published = zarr.open_group(str(root), mode="a")

    # First-import case: the prepared transaction also owns the coordinates.
    meta_path = root / CUBE_DIR / "CUBE_META.json"
    if not meta_path.is_file():
        published.create_dataset(
            "coords/latitude",
            data=np.asarray(stage["coords/latitude"][:], dtype="float64"),
            overwrite=False)
        published.create_dataset(
            "coords/longitude",
            data=np.asarray(stage["coords/longitude"][:], dtype="float64"),
            overwrite=False)
        lat_n = int(published["coords/latitude"].shape[0])
        lon_n = int(published["coords/longitude"].shape[0])
        meta = {
            "format": "grib-assembler-cube/v1",
            "created_at": None,
            "grid_digest": journal.get("initial_grid_digest"),
            "grid_shape": [lat_n, lon_n],
            "storage": "zarr-v2",
            "recovered": True,
        }
        _atomic_write_json(meta_path, meta)

    # Publish the frozen registry snapshot.
    for name, doc in snapshot.items():
        _atomic_write_json(registry_dir / name, doc)
    origins = list(snapshot["origins.json"]["revisions"])
    oi = int(journal["origin_index"])
    nj = int(published["coords/latitude"].shape[0])
    ni = int(published["coords/longitude"].shape[0])

    # Coordinate extents from the frozen snapshot.
    n_levels = len(snapshot["levels.json"])
    n_members = len(snapshot["members.json"])
    n_slots = len(snapshot["slots.json"])
    shape6 = (len(origins), n_levels, n_members, n_slots, nj, ni)
    shape4 = shape6[:4]

    # Grow every staged variable's published arrays to the global shape first
    # (mirrors CubeStore._merge_cells so absent variables keep missing slabs).
    staged_var_names = list(stage.get("vars", {}))
    for var_key_name in staged_var_names:
        _reopen_or_create(published, var_key_name, "values", shape6,
                          dtype="float64", fill=np.nan)
        _reopen_or_create(published, var_key_name, "present", shape4,
                          dtype="uint8", fill=0)
        _reopen_or_create(published, var_key_name, "message_digest", shape4,
                          dtype=DIGEST_DTYPE, fill="")
        _reopen_or_create(published, var_key_name, "package_digest", shape4,
                          dtype=DIGEST_DTYPE, fill="")

    # Replay the per-cell merge from staging.
    for entry in journal.get("written_cells", []):
        var_key_name, ci, li, mi, si = entry
        idx4 = (ci, li, mi, si)
        s_present = stage[f"vars/{var_key_name}/present"]
        if int(s_present[idx4]) != 1:
            continue
        values = _reopen_or_create(published, var_key_name, "values",
                                   shape6, dtype="float64", fill=np.nan)
        present = _reopen_or_create(published, var_key_name, "present",
                                    shape4, dtype="uint8", fill=0)
        msg_dig = _reopen_or_create(published, var_key_name,
                                    "message_digest", shape4,
                                    dtype=DIGEST_DTYPE, fill="")
        pkg_dig = _reopen_or_create(published, var_key_name,
                                    "package_digest", shape4,
                                    dtype=DIGEST_DTYPE, fill="")
        values[idx4] = np.asarray(
            stage[f"vars/{var_key_name}/values"][idx4], dtype="float64")
        present[idx4] = np.uint8(1)
        msg_dig[idx4] = str(stage[
            f"vars/{var_key_name}/message_digest"][idx4])
        pkg_dig[idx4] = str(stage[
            f"vars/{var_key_name}/package_digest"][idx4])

    marker = {
        "origin": origin,
        "origin_index": oi,
        "package_digest": journal["package_digest"],
        "txn_id": journal["txn_id"],
        "committed_at": None,
        "recovered": True,
        "flushed_cells": journal.get("flushed_cells"),
    }
    marker_path = (root / CUBE_DIR / COMMIT_DIR
                   / f"{marker_name(origin)}.json")
    tmp = marker_path.with_name(marker_path.name + ".tmp")
    tmp.write_text(json.dumps(marker, indent=2, sort_keys=True),
                   encoding="utf-8")
    os.replace(tmp, marker_path)
    journal["state"] = "committed"
    _atomic_write_json(txn_dir / "journal.json", journal)
    shutil.rmtree(txn_dir, ignore_errors=True)
    report.resumed_transactions.append(txn_dir.name)


def _reconcile_registry(root: Path, report: RecoveryReport) -> None:
    """Drop registry origins that never acquired a commit marker.

    Appends are strictly ordered, so an uncommitted origin is always the
    trailing index; the zarr slabs for it are truncated and the coordinate
    registries are rewritten without it.  Committed origins are untouched.
    Arrays may be one resize ahead of the registry JSON (a crash between the
    array merge and the registry swap); they are truncated to the committed
    length in that case too.
    """

    reg_dir = root / CUBE_DIR / "registry"
    origins_doc = _load_json(reg_dir / "origins.json", {"revisions": []})
    origins = list(origins_doc.get("revisions", []))
    committed = set(_committed_origins(root))
    orphaned = [o for o in origins if o not in committed]
    keep = [o for o in origins if o in committed]
    if orphaned and orphaned != origins[len(keep):]:
        raise CubeError("non_trailing_orphan_origin", {
            "orphaned": orphaned, "origins": origins})

    # Shrink every published array's origin axis to the committed length.
    vars_dir = root / "vars"
    if vars_dir.exists():
        group = zarr.open_group(str(root), mode="a")
        for var_key_name in list(group.get("vars", {})):
            for leaf in ("values", "present", "message_digest",
                         "package_digest"):
                arr = group[f"vars/{var_key_name}/{leaf}"]
                if arr.shape[0] > len(committed):
                    arr.resize((len(committed),) + tuple(arr.shape[1:]))

    if not orphaned:
        return

    _atomic_write_json(reg_dir / "origins.json", {"revisions": keep})
    sources = _load_json(reg_dir / "sources.json", {})
    for o in orphaned:
        sources.pop(o, None)
    _atomic_write_json(reg_dir / "sources.json", sources)
    report.orphaned_origins.extend(orphaned)
