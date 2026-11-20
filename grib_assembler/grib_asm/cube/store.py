"""Transactional Zarr storage for the analysis cube.

Design goals (mapped directly to the acceptance rules):

* **per-cycle commits.** One import of one hand-off package = one
  transaction for one reference time.  Readers can only see cycles whose
  commit marker exists; coordinates are never visible without their arrays.
* **crash safety without a database.** All work for a cycle happens under
  ``.cube/txn/<txn_id>/`` as an independent zarr group plus a transaction
  journal.  Publication ends with a single ``os.replace`` of the commit
  marker (atomic on one filesystem); the zarr group root itself is only
  opened once the marker exists.
* **no rewrite of committed cycles.** Coordinate registries live in
  append-only JSON sidecars (``.cube/registry/``); committing a cycle writes
  new sidecar versions and swaps them atomically, leaving the previous
  revision on disk for recovery/audit.  Array slabs of committed origins are
  never reopened for writing.
* **bounded memory.** Values are written per message in configurable
  batches; each zarr array uses chunk shapes aligned to analysis slices
  (one full grid per chunk, scalar chunks along the other dimensions).

The zarr layout (zarr v2 storage spec)::

    <store>/
      .cube/
        CUBE_META.json              # dataset-level grid/identity info
        registry/
          origins.json              #   committed reference times, ordered
          variables.json            #   variable index + per-variable metadata
          levels.json               #   level index
          members.json              #   member index
          slots.json                #   time-slot index
          sources.json              #   origin -> package digest + evidence
          conflicts.json            # retained source conflicts
          committed/<origin>.json   # per-cycle commit markers
        conflicts/<origin>__<digest>.json
        txn/<txn_id>/               # uncommitted work (journal + staging)
      coords/
        latitude/<zarr array>       # (Nj,)
        longitude/<zarr array>      # (Ni,)
      vars/<var_key>/
        values/<zarr array>         # float64 (No, Nl, Nm, Ns, Nj, Ni)
        present/<zarr array>        # uint8  1 = present, 0 = explicit missing
        message_digest/<zarr array> # fixed-width unicode, per cell source
        package_digest/<zarr array> # fixed-width unicode, per cell source
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..timeutil import iso
from .coords import (
    level_doc, level_key, member_doc, member_key, slot_doc, slot_key,
    variable_doc, variable_key,
)
from .errors import CubeError

CUBE_DIR = ".cube"
REGISTRY_DIR = "registry"
COMMIT_DIR = "committed"
TXN_DIR = "txn"
CONFLICT_DIR = "conflicts"
META_FILE = "CUBE_META.json"
# 64 hex chars of SHA-256; store wider to avoid future truncation surprises.
DIGEST_DTYPE = "U80"


def _now() -> str:
    return iso(datetime.now(timezone.utc))


def _atomic_write_json(path: Path, doc) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=True,
                              ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _load_json(path: Path, default):
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def marker_name(origin: str) -> str:
    return origin.replace(":", "").replace("-", "")


@dataclass
class CubeConfig:
    """Tunables bounding memory and chunk geometry."""

    # Maximum number of messages decoded/held before a flush to zarr.
    batch_size: int = 16
    # Chunk shape of the values array along (origin, level, member, slot).
    # The spatial axes always use one full-grid chunk (Nj, Ni).
    coord_chunk: tuple[int, int, int, int] = (1, 1, 1, 1)
    compressor_name: str = "zstd"
    compressor_level: int = 3

    def validate(self) -> None:
        if self.batch_size < 1:
            raise CubeError("invalid_config", {"batch_size": self.batch_size})
        if len(self.coord_chunk) != 4 or any(
                c < 1 for c in self.coord_chunk):
            raise CubeError("invalid_config",
                            {"coord_chunk": list(self.coord_chunk)})


class Registry:
    """In-memory view of the append-only coordinate registries."""

    def __init__(self, root: Path):
        self.root = root
        self.reg_dir = root / CUBE_DIR / REGISTRY_DIR
        origins_doc = _load_json(self.reg_dir / "origins.json",
                                 {"revisions": []})
        self.origins: list[str] = list(origins_doc.get("revisions", []))
        self.variables: dict = _load_json(
            self.reg_dir / "variables.json", {})
        self.levels: dict = _load_json(self.reg_dir / "levels.json", {})
        self.members: dict = _load_json(self.reg_dir / "members.json", {})
        self.slots: dict = _load_json(self.reg_dir / "slots.json", {})
        # Absolute UTC validity boundaries are per origin: the same relative
        # slot (e.g. instant_0) has different absolute bounds at 00Z and 12Z.
        self.slot_validity_map: dict = _load_json(
            self.reg_dir / "slot_validity.json", {})
        self.sources: dict = _load_json(
            self.reg_dir / "sources.json", {})
        conflicts_doc = _load_json(self.reg_dir / "conflicts.json",
                                   {"entries": []})
        self.conflicts: list = list(conflicts_doc.get("entries", []))

    # ------------------------------------------------------------- lookups
    def origin_index(self, origin: str) -> int | None:
        return self.origins.index(origin) if origin in self.origins else None

    def variable_index(self, var) -> int | None:
        entry = self.variables.get(variable_key(var))
        return None if entry is None else entry["index"]

    def level_index(self, level) -> int | None:
        entry = self.levels.get(level_key(level))
        return None if entry is None else entry["index"]

    def member_index(self, member) -> int | None:
        entry = self.members.get(member_key(member))
        return None if entry is None else entry["index"]

    def slot_index(self, slot) -> int | None:
        entry = self.slots.get(slot_key(slot))
        return None if entry is None else entry["index"]

    # ------------------------------------------------------------ mutation
    def ensure_variable(self, var, unit_info, extra_attrs: dict) -> str:
        key = variable_key(var)
        if key not in self.variables:
            self.variables[key] = {
                "index": len(self.variables),
                "key": key,
                "identity": variable_doc(var),
                "units": unit_info.doc(),
                "attrs": extra_attrs,
                "registered_at": _now(),
            }
        return key

    def ensure_level(self, level) -> str:
        key = level_key(level)
        if key not in self.levels:
            self.levels[key] = {
                "index": len(self.levels),
                "key": key,
                "identity": level_doc(level),
                "registered_at": _now(),
            }
        return key

    def ensure_member(self, member) -> str:
        key = member_key(member)
        if key not in self.members:
            self.members[key] = {
                "index": len(self.members),
                "key": key,
                "identity": member_doc(member),
                "registered_at": _now(),
            }
        return key

    def ensure_slot(self, slot) -> str:
        """Register the relative time-slot identity (origin-independent)."""

        key = slot_key(slot)
        if key not in self.slots:
            self.slots[key] = {
                "index": len(self.slots),
                "key": key,
                "identity": slot_doc(slot),
                "is_instant": slot.is_instant,
                "registered_at": _now(),
            }
        return key

    def set_slot_validity(self, origin: str, slot, validity_start: str,
                          validity_end: str) -> None:
        """Record absolute UTC validity bounds for one origin's slot."""

        key = slot_key(slot)
        per_origin = self.slot_validity_map.setdefault(origin, {})
        per_origin[key] = {
            "validity_start": validity_start,
            "validity_end": validity_end,
        }

    def ensure_origin(self, origin: str) -> int:
        if origin not in self.origins:
            self.origins.append(origin)
        return self.origins.index(origin)

    # ------------------------------------------------------------- persist
    def snapshot_docs(self) -> dict[str, dict]:
        return {
            "origins.json": {"revisions": self.origins},
            "variables.json": self.variables,
            "levels.json": self.levels,
            "members.json": self.members,
            "slots.json": self.slots,
            "slot_validity.json": self.slot_validity_map,
            "sources.json": self.sources,
            "conflicts.json": {"entries": self.conflicts},
        }

    def write(self, reg_dir: Path | None = None) -> None:
        target = reg_dir or self.reg_dir
        for name, doc in self.snapshot_docs().items():
            _atomic_write_json(target / name, doc)


@dataclass
class PendingCell:
    """One decoded cell staged for a batched flush."""

    var_key: str
    origin_index: int
    level_index: int
    member_index: int
    slot_index: int
    values: np.ndarray
    message_digest: str
    package_digest: str


@dataclass
class Transaction:
    txn_id: str
    txn_dir: Path
    origin: str
    origin_index: int
    package_digest: str
    journal: dict
    initial_grid: object | None = None
    initial_grid_digest: str | None = None
    # Every cell already flushed, as (var_key, oi, li, mi, si); this lets the
    # commit/recovery copy one grid at a time instead of whole origin slabs.
    written_cells: list[tuple] = field(default_factory=list)


class CubeStore:
    """Opens/creates the dataset and manages per-cycle transactions."""

    def __init__(self, store_path: str | Path, config: CubeConfig | None = None):
        self.root = Path(store_path)
        self.config = config or CubeConfig()
        self.config.validate()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / CUBE_DIR / REGISTRY_DIR).mkdir(parents=True,
                                                    exist_ok=True)
        (self.root / CUBE_DIR / COMMIT_DIR).mkdir(parents=True, exist_ok=True)
        (self.root / CUBE_DIR / TXN_DIR).mkdir(parents=True, exist_ok=True)
        (self.root / CUBE_DIR / CONFLICT_DIR).mkdir(parents=True,
                                                    exist_ok=True)
        self._meta_path = self.root / CUBE_DIR / META_FILE
        self.meta = _load_json(self._meta_path, None)
        import zarr
        import numcodecs
        self._zarr = zarr
        self._numcodecs = numcodecs

    # ---------------------------------------------------------------- zarr
    def _compressor(self):
        try:
            return self._numcodecs.Zstd(level=self.config.compressor_level)
        except Exception:
            # Zstd unavailable in a stripped build: fall back to the stable
            # default rather than fail dataset creation.
            return self._numcodecs.Blosc(
                cname="lz4", clevel=self.config.compressor_level)

    def is_initialized(self) -> bool:
        return self.meta is not None

    # ----------------------------------------------------------- transactions
    def begin_transaction(self, origin: str, package_digest: str,
                          registry: Registry, *,
                          initial_grid=None,
                          initial_grid_digest: str | None = None) -> Transaction:
        txn_id = (f"{marker_name(origin)}__{uuid.uuid4().hex[:12]}")
        txn_dir = self.root / CUBE_DIR / TXN_DIR / txn_id
        txn_dir.mkdir(parents=True, exist_ok=False)
        origin_index = registry.ensure_origin(origin)
        journal = {
            "txn_id": txn_id,
            "state": "open",
            "origin": origin,
            "origin_index": origin_index,
            "package_digest": package_digest,
            "created_at": _now(),
            "flushed_cells": 0,
            "written_cells": [],
            "files": [],
        }
        txn = Transaction(
            txn_id=txn_id, txn_dir=txn_dir, origin=origin,
            origin_index=origin_index, package_digest=package_digest,
            journal=journal, initial_grid=initial_grid,
            initial_grid_digest=initial_grid_digest)

        # First-ever import: coordinates are staged too, and only reach the
        # published root at commit time.
        if initial_grid is not None:
            stage = self._zarr.open_group(str(txn_dir / "stage"), mode="a")
            compressor = self._compressor()
            stage.create_dataset(
                "coords/latitude",
                data=initial_grid.latitude.astype("float64"),
                chunks=initial_grid.latitude.shape, overwrite=False,
                compressor=compressor)
            stage.create_dataset(
                "coords/longitude",
                data=initial_grid.longitude.astype("float64"),
                chunks=initial_grid.longitude.shape, overwrite=False,
                compressor=compressor)
        _atomic_write_json(txn_dir / "journal.json", journal)
        return txn

    def flush(self, txn: Transaction, cells: list[PendingCell],
              registry: Registry) -> None:
        """Write a batch of decoded cells into the staging zarr group.

        Arrays are created lazily per variable and grown as new coordinate
        entries appear.  Each write targets exactly one scalar-coordinate
        chunk plus the full-grid spatial chunk, so memory and I/O stay aligned
        to the analysis slice shape.
        """

        if not cells:
            return
        stage = self._zarr.open_group(str(txn.txn_dir / "stage"), mode="a")
        nj, ni = self._grid_shape(txn)
        oc, lc, mc, sc = self.config.coord_chunk
        compressor = self._compressor()

        by_var: dict[str, list[PendingCell]] = {}
        for cell in cells:
            by_var.setdefault(cell.var_key, []).append(cell)

        for var_key_name, var_cells in by_var.items():
            n_orig = len(registry.origins)
            n_lvl = len(registry.levels)
            n_mem = len(registry.members)
            n_slot = len(registry.slots)
            shape6 = (n_orig, n_lvl, n_mem, n_slot, nj, ni)
            shape4 = (n_orig, n_lvl, n_mem, n_slot)
            chunks6 = (oc, lc, mc, sc, nj, ni)
            chunks4 = (oc, lc, mc, sc)
            values = self._require_array(
                stage, f"vars/{var_key_name}/values", shape=shape6,
                chunks=chunks6, dtype="float64", fill_value=np.nan,
                compressor=compressor)
            present = self._require_array(
                stage, f"vars/{var_key_name}/present", shape=shape4,
                chunks=chunks4, dtype="uint8", fill_value=0,
                compressor=compressor)
            msg_dig = self._require_array(
                stage, f"vars/{var_key_name}/message_digest", shape=shape4,
                chunks=chunks4, dtype=DIGEST_DTYPE, fill_value="",
                compressor=compressor)
            pkg_dig = self._require_array(
                stage, f"vars/{var_key_name}/package_digest", shape=shape4,
                chunks=chunks4, dtype=DIGEST_DTYPE, fill_value="",
                compressor=compressor)
            for cell in var_cells:
                idx4 = (cell.origin_index, cell.level_index,
                        cell.member_index, cell.slot_index)
                values[idx4] = cell.values
                present[idx4] = np.uint8(1)
                msg_dig[idx4] = cell.message_digest
                pkg_dig[idx4] = cell.package_digest
                txn.written_cells.append((cell.var_key,) + idx4)
                txn.journal["flushed_cells"] += 1
        txn.journal["written_cells"] = [list(c) for c in txn.written_cells]
        _atomic_write_json(txn.txn_dir / "journal.json", txn.journal)

    def _grid_shape(self, txn: Transaction) -> tuple[int, int]:
        if self.meta is not None:
            return tuple(self.meta["grid_shape"])
        if txn.initial_grid is not None:
            return (int(txn.initial_grid.nj), int(txn.initial_grid.ni))
        raise CubeError("grid_shape_unknown", {"txn_id": txn.txn_id})

    def _require_array(self, group, path: str, *, shape, chunks, dtype,
                       fill_value, compressor):
        """Open an array or create it; grow it when coordinates were added."""

        existing = group.get(path)
        if existing is None:
            return group.create_dataset(
                path, shape=shape, chunks=chunks, dtype=dtype,
                fill_value=fill_value, compressor=compressor,
                overwrite=False)
        if tuple(existing.shape) != tuple(shape):
            existing.resize(shape)
        return existing

    def prepare(self, txn: Transaction, registry: Registry,
                source_doc: dict) -> None:
        """Freeze the journal and snapshot the registry for the swap."""

        snap_dir = txn.txn_dir / "registry_snapshot"
        snap_dir.mkdir(exist_ok=True)
        registry.write(snap_dir)
        _atomic_write_json(txn.txn_dir / "source.json", source_doc)
        txn.journal["state"] = "prepared"
        txn.journal["prepared_at"] = _now()
        _atomic_write_json(txn.txn_dir / "journal.json", txn.journal)

    def commit(self, txn: Transaction, registry: Registry,
               fail_point: str | None = None) -> None:
        """Publish the cycle.

        Order matters: staged chunks are copied first, coordinate registries
        are swapped second, and the per-origin commit marker is the last
        ``os.replace``.  A crash at any earlier step leaves no half-visible
        cycle; recovery truncates the trailing uncommitted origin.
        """

        if txn.journal["state"] != "prepared":
            raise CubeError("transaction_not_prepared",
                            {"txn_id": txn.txn_id})
        root_group = self._zarr.open_group(str(self.root), mode="a")
        nj, ni = self._grid_shape(txn)

        # 0. First import: publish coordinates + dataset metadata.
        if txn.initial_grid is not None:
            stage = self._zarr.open_group(str(txn.txn_dir / "stage"),
                                          mode="r")
            compressor = self._compressor()
            root_group.create_dataset(
                "coords/latitude",
                data=np.asarray(stage["coords/latitude"][:],
                                dtype="float64"),
                chunks=(nj,), overwrite=False, compressor=compressor)
            root_group.create_dataset(
                "coords/longitude",
                data=np.asarray(stage["coords/longitude"][:],
                                dtype="float64"),
                chunks=(ni,), overwrite=False, compressor=compressor)
            meta = {
                "format": "grib-assembler-cube/v1",
                "created_at": _now(),
                "grid_digest": txn.initial_grid_digest,
                "grid_definition": txn.initial_grid.definition_doc(),
                "grid_shape": [nj, ni],
                "storage": "zarr-v2",
            }
            _atomic_write_json(self._meta_path, meta)
            self.meta = meta
        if fail_point == "during_array_write":
            _hard_exit()

        # 1. Merge staged arrays one cell (one full grid) at a time.
        self._merge_cells(txn, registry)
        if fail_point == "during_registry_swap":
            _hard_exit()

        # 2. Swap the coordinate registries atomically.
        registry.write()
        if fail_point == "during_commit_marker":
            _hard_exit()

        # 3. Publish the per-origin commit marker last.
        marker = {
            "origin": txn.origin,
            "origin_index": txn.origin_index,
            "package_digest": txn.package_digest,
            "txn_id": txn.txn_id,
            "committed_at": _now(),
            "flushed_cells": txn.journal["flushed_cells"],
        }
        marker_path = (self.root / CUBE_DIR / COMMIT_DIR
                       / f"{marker_name(txn.origin)}.json")
        tmp = marker_path.with_name(marker_path.name + ".tmp")
        tmp.write_text(json.dumps(marker, indent=2, sort_keys=True),
                       encoding="utf-8")
        os.replace(tmp, marker_path)
        txn.journal["state"] = "committed"
        _atomic_write_json(txn.txn_dir / "journal.json", txn.journal)

    def _merge_cells(self, txn: Transaction, registry: Registry) -> None:
        """Copy only this origin's cells; committed slabs stay untouched."""

        stage = self._zarr.open_group(str(txn.txn_dir / "stage"), mode="r")
        root_group = self._zarr.open_group(str(self.root), mode="a")
        nj, ni = self._grid_shape(txn)
        oc, lc, mc, sc = self.config.coord_chunk
        shape6 = (len(registry.origins), len(registry.levels),
                  len(registry.members), len(registry.slots), nj, ni)
        shape4 = shape6[:4]

        # Grow *every* published variable's arrays to the global shape first:
        # variables absent from this package keep an explicit missing slab
        # for the new origin rather than a stale, too-short origin axis.
        all_var_names = set(root_group.get("vars", {}).keys())
        for var_key_name in all_var_names:
            for leaf, shape, dtype, fill in (
                ("values", shape6, "float64", np.nan),
                ("present", shape4, "uint8", 0),
                ("message_digest", shape4, DIGEST_DTYPE, ""),
                ("package_digest", shape4, DIGEST_DTYPE, "")):
                self._require_array(
                    root_group, f"vars/{var_key_name}/{leaf}",
                    shape=shape,
                    chunks=(oc, lc, mc, sc, nj, ni)
                    if leaf == "values" else (oc, lc, mc, sc),
                    dtype=dtype, fill_value=fill,
                    compressor=self._compressor())

        var_names = {c[0] for c in txn.written_cells}
        arrays: dict[str, dict] = {}
        for var_key_name in var_names:
            bundle = {}
            bundle["values"] = self._require_array(
                root_group, f"vars/{var_key_name}/values", shape=shape6,
                chunks=(oc, lc, mc, sc, nj, ni), dtype="float64",
                fill_value=np.nan, compressor=self._compressor())
            bundle["present"] = self._require_array(
                root_group, f"vars/{var_key_name}/present", shape=shape4,
                chunks=(oc, lc, mc, sc), dtype="uint8", fill_value=0,
                compressor=self._compressor())
            bundle["message_digest"] = self._require_array(
                root_group, f"vars/{var_key_name}/message_digest",
                shape=shape4, chunks=(oc, lc, mc, sc), dtype=DIGEST_DTYPE,
                fill_value="", compressor=self._compressor())
            bundle["package_digest"] = self._require_array(
                root_group, f"vars/{var_key_name}/package_digest",
                shape=shape4, chunks=(oc, lc, mc, sc), dtype=DIGEST_DTYPE,
                fill_value="", compressor=self._compressor())
            arrays[var_key_name] = bundle

        for entry in txn.written_cells:
            var_key_name, oi, li, mi, si = entry
            idx4 = (oi, li, mi, si)
            s_values = stage[f"vars/{var_key_name}/values"]
            s_present = stage[f"vars/{var_key_name}/present"]
            s_msg = stage[f"vars/{var_key_name}/message_digest"]
            s_pkg = stage[f"vars/{var_key_name}/package_digest"]
            if int(s_present[idx4]) != 1:
                continue
            bundle = arrays[var_key_name]
            # One full-grid array at a time: memory stays bounded by the
            # single-message grid size, never by the whole dataset.
            bundle["values"][idx4] = np.asarray(s_values[idx4],
                                                dtype="float64")
            bundle["present"][idx4] = np.uint8(1)
            bundle["message_digest"][idx4] = str(s_msg[idx4])
            bundle["package_digest"][idx4] = str(s_pkg[idx4])

    def rollback(self, txn: Transaction) -> None:
        if txn.txn_dir.exists():
            shutil.rmtree(txn.txn_dir)


def _hard_exit() -> None:
    """Simulate a hard process kill (fault injection)."""

    os._exit(42)
