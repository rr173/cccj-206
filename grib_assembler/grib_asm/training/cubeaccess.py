"""Bounded, provenance-annotated point access over the committed Zarr cube.

The export reads only *committed* cube state and only through this module.
Three guarantees matter for the acceptance rules:

1. **a values block is decoded at most once per successful export.**  Point
   requests are grouped on disk by their (variable, origin, level, member,
   slot) chunk first; one chunk is then read, all requested grid points are
   extracted from that single in-memory grid, and the grid is released before
   the next chunk.  A wrapper store counts every physical ``values`` key
   fetch, giving run evidence of the read count.
2. **requests that fail station/publish/field checks never reach here**, so
   no value block is decoded for a cell that cannot be used.
3. **every read cell returns its full evidence** - present mask, per-cell
   message digest, package digest and the raw GRIB unit - even when the grid
   point itself is explicitly missing.  Digest/unit disagreements with the
   cube's committed source are treated as evidence errors, never as data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..cube.coords import level_key, member_key, slot_key, variable_key
from ..cube.query import CubeReader
from ..cube.store import CUBE_DIR
from .errors import CubeEvidenceError
from .model import FeatureColumn

try:  # zarr v2 store base class
    from zarr._storage.store import BaseStore
except Exception:  # pragma: no cover - depends on zarr packaging
    from zarr.storage import BaseStore


class CountingStore(BaseStore):
    """Transparent store wrapper counting physical value-block fetches.

    Only keys belonging to a ``values`` zarr array and naming an actual chunk
    (not the ``.zarray`` metadata) increment the counter, so the number is
    directly comparable to the number of distinct analysis slices decoded.
    """

    def __init__(self, inner):
        self.inner = inner
        self.value_chunk_gets: list[str] = []
        self.other_gets = 0

    @staticmethod
    def _is_value_chunk(key: str) -> bool:
        parts = key.split("/")
        return (len(parts) >= 2 and parts[-2] == "values"
                and not parts[-1].startswith(".z"))

    def __getitem__(self, key):
        if self._is_value_chunk(key):
            self.value_chunk_gets.append(key)
        else:
            self.other_gets += 1
        return self.inner[key]

    def __setitem__(self, key, value):
        self.inner[key] = value

    def __delitem__(self, key):
        del self.inner[key]

    def __contains__(self, key):
        return key in self.inner

    def __iter__(self):
        return iter(self.inner)

    def __len__(self):
        return len(self.inner)

    def keys(self):
        return self.inner.keys()


@dataclass
class CellEvidence:
    """Everything the output row records about one resolved feature cell."""

    present: bool
    value: float | None
    message_digest: str
    package_digest: str
    raw_unit: str
    grid_lat_index: int
    grid_lon_index: int


@dataclass
class ChunkCoordinates:
    """Resolved registry indices for one feature column at one origin."""

    var_key: str
    origin_index: int
    level_index: int
    member_index: int
    slot_index: int


class CubePointReader:
    """Read-only, instrumented point reader over one frozen cube store."""

    def __init__(self, store_path: str | Path):
        self.root = Path(store_path)
        # Build the counting view first, then let CubeReader open zarr on it.
        import zarr
        base_group = zarr.open_group(str(self.root), mode="r")
        self.counting_store = CountingStore(base_group.store)
        self.reader = CubeReader.__new__(CubeReader)
        self.reader.root = self.root
        meta_path = self.root / CUBE_DIR / "CUBE_META.json"
        if not meta_path.is_file():
            raise FileNotFoundError(
                f"not an initialized cube store: {self.root}")
        self.reader.meta = json.loads(
            meta_path.read_text(encoding="utf-8"))
        from ..cube.store import Registry
        self.reader.registry = Registry(self.root)
        self.reader._group = zarr.open_group(
            self.counting_store, mode="r")
        from ..cube.recovery import marker_exists
        committed = [o for o in self.reader.registry.origins
                     if marker_exists(self.root, o)]
        self.reader._committed_set = set(committed)
        self.reader._committed_origins = committed
        self.grid_digest = self.reader.meta["grid_digest"]
        self.grid_shape = tuple(self.reader.meta["grid_shape"])  # (nj, ni)
        self.nj, self.ni = self.grid_shape
        self._cell_meta_cache: dict[tuple, dict] = {}
        self._decoded_chunk_cache: dict[tuple, np.ndarray] = {}

    # ------------------------------------------------------------ lookups
    def is_origin_committed(self, origin_iso: str) -> bool:
        return origin_iso in self.reader._committed_set

    def source_digest_of(self, origin_iso: str) -> str | None:
        source = self.reader.registry.sources.get(origin_iso)
        return None if source is None else source.get("package_digest")

    def resolve_coordinates(self, column: FeatureColumn,
                            origin_iso: str) -> ChunkCoordinates | None:
        """Map a feature column + origin to registry indices.

        Returns ``None`` (a normal ``field_missing`` gap) only when the
        variable itself is unknown to the cube or the origin is uncommitted;
        the caller has already validated the origin before calling.
        """

        registry = self.reader.registry
        vk = variable_key(column.variable)
        if vk not in registry.variables:
            return None
        if origin_iso not in self.reader._committed_set:
            return None
        level_entry = registry.levels.get(level_key(column.level))
        member_entry = registry.members.get(member_key(column.member))
        slot_entry = registry.slots.get(slot_key(column.slot))
        if level_entry is None or member_entry is None or slot_entry is None:
            # A published package cannot reference a level/member/slot the
            # cube never registered: that is contradictory evidence.
            raise CubeEvidenceError("cell_identity_not_registered", {
                "variable_key": vk, "origin": origin_iso,
                "level": column.level.to_dict(),
                "member": column.member.to_dict(),
                "slot": column.slot.to_dict()})
        return ChunkCoordinates(
            var_key=vk,
            origin_index=registry.origin_index(origin_iso),
            level_index=level_entry["index"],
            member_index=member_entry["index"],
            slot_index=slot_entry["index"])

    def variable_unit_info(self, var_key: str) -> dict:
        return self.reader.registry.variables[var_key]["units"]

    def validate_grid_point(self, grid_digest: str, lat_index: int,
                            lon_index: int) -> None:
        """Reject maps onto another grid or out-of-bounds axis indices."""

        if grid_digest != self.grid_digest:
            raise CubeEvidenceError("station_grid_mismatch", {
                "map_grid_digest": grid_digest,
                "cube_grid_digest": self.grid_digest})
        if not (0 <= lat_index < self.nj and 0 <= lon_index < self.ni):
            raise CubeEvidenceError("station_index_out_of_bounds", {
                "lat_index": lat_index, "lon_index": lon_index,
                "grid_shape": [self.nj, self.ni]})

    # ------------------------------------------------------------- reading
    def cell_metadata(self, coords: ChunkCoordinates) -> dict:
        """Scalar-coordinate evidence (mask + digests); cached per cell.

        The mask/digest arrays use scalar chunks, so checking whether a field
        exists never decodes a ``values`` block.  The cache keeps repeated
        samples of the same cell at one scalar read each.
        """

        token = (coords.var_key, coords.origin_index, coords.level_index,
                 coords.member_index, coords.slot_index)
        cached = self._cell_meta_cache.get(token)
        if cached is not None:
            return cached
        group = self.reader._group
        idx4 = (coords.origin_index, coords.level_index,
                coords.member_index, coords.slot_index)
        present_arr = group[f"vars/{coords.var_key}/present"]
        present = int(present_arr[idx4]) == 1
        meta = {"present": present, "message_digest": "",
                "package_digest": "", "raw_unit": ""}
        if present:
            meta["message_digest"] = str(
                group[f"vars/{coords.var_key}/message_digest"][idx4])
            meta["package_digest"] = str(
                group[f"vars/{coords.var_key}/package_digest"][idx4])
            meta["raw_unit"] = self.variable_unit_info(
                coords.var_key)["raw_unit"]
        self._cell_meta_cache[token] = meta
        return meta

    def read_chunk_values(self, coords: ChunkCoordinates) -> np.ndarray:
        """Decode exactly one full-grid values chunk, once, then cache it.

        The cache holds at most one grid at a time in the pipeline (the
        exporter clears it between chunks), bounding memory by one grid plus
        the buffered request batch.
        """

        token = (coords.var_key, coords.origin_index, coords.level_index,
                 coords.member_index, coords.slot_index)
        cached = self._decoded_chunk_cache.get(token)
        if cached is not None:
            return cached
        values_arr = self.reader._group[f"vars/{coords.var_key}/values"]
        grid = np.asarray(values_arr[
            coords.origin_index, coords.level_index,
            coords.member_index, coords.slot_index, :, :], dtype="float64")
        self._decoded_chunk_cache[token] = grid
        return grid

    def clear_chunk_cache(self) -> None:
        self._decoded_chunk_cache.clear()

    def point_from_chunk(self, coords: ChunkCoordinates,
                         lat_index: int, lon_index: int) -> float:
        grid = self.read_chunk_values(coords)
        value = grid[lat_index, lon_index]
        # NaN survives JSON as null at write time.
        return None if np.isnan(value) else float(value)

    # ---------------------------------------------------------- diagnostics
    def value_block_read_count(self) -> int:
        return len(self.counting_store.value_chunk_gets)

    def value_block_read_keys(self) -> list[str]:
        return list(self.counting_store.value_chunk_gets)
