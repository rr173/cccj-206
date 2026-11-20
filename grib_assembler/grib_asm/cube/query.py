"""Stable, slice-based read access to a published analysis cube.

The reader only exposes *committed* coordinates (origin entries with a
commit marker).  Every query returns the values together with their full
coordinate labels, the explicit presence mask (never inferred from NaN
alone), and per-cell provenance: the originating hand-off package digest
and the exact source-message SHA-256.

Selection is by the same numeric identities used at import time:

* variable: full GRIB2 identity (short names are never used to merge or
  select variables);
* level: exact fixed-surface type/rational;
* member: deterministic/control/perturbed kind and number;
* time: validity-time interval on the absolute UTC boundary, or relative
  start/end seconds.

Result order is deterministic: coordinates come back in registry order and
cell rows are sorted by (origin, level, member, slot) indices.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..identity import Level, MemberId, VariableId
from ..timeutil import TimeSlot
from .coords import level_key, member_key, slot_key, variable_key
from .store import CUBE_DIR, Registry, marker_name
from .recovery import marker_exists


@dataclass
class SliceResult:
    """One query result: values plus coordinates, mask and provenance."""

    variable_identity: dict
    units: dict
    axes: dict                       # axis name -> ordered coordinate labels
    values: np.ndarray               # float64, NaN where not present
    present: np.ndarray              # uint8 mask, shape of coord axes (4-D)
    message_digest: np.ndarray       # U80 provenance, shape of coord axes
    package_digest: np.ndarray       # U80 provenance, shape of coord axes
    selected: dict                   # the selection filters applied

    def to_dict(self) -> dict:
        return {
            "variable_identity": self.variable_identity,
            "units": self.units,
            "axes": self.axes,
            "selected": self.selected,
            "present_shape": list(self.present.shape),
            "present_count": int(np.count_nonzero(self.present)),
            "missing_count": int(self.present.size
                                 - np.count_nonzero(self.present)),
        }


class CubeReader:
    """Read-only view over a committed cube."""

    def __init__(self, store_path: str | Path):
        self.root = Path(store_path)
        meta_path = self.root / CUBE_DIR / "CUBE_META.json"
        if not meta_path.is_file():
            raise FileNotFoundError(
                f"not an initialized cube store: {self.root}")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.registry = Registry(self.root)
        import zarr
        self._group = zarr.open_group(str(self.root), mode="r")
        self._hide_uncommitted_origins()

    # ------------------------------------------------------------ invariants
    def _hide_uncommitted_origins(self) -> None:
        committed = [o for o in self.registry.origins
                     if marker_exists(self.root, o)]
        self._committed_set = set(committed)
        self._committed_origins = committed

    @property
    def grid(self) -> dict:
        return dict(self.meta["grid_definition"])

    def origins(self) -> list[str]:
        return list(self._committed_origins)

    def variables(self) -> list[dict]:
        return [self.registry.variables[k]
                for k in sorted(self.registry.variables,
                                key=lambda kk: self.registry.variables[kk][
                                    "index"])]

    def levels(self) -> list[dict]:
        return [self.registry.levels[k]
                for k in sorted(self.registry.levels,
                                key=lambda kk: self.registry.levels[kk][
                                    "index"])]

    def members(self) -> list[dict]:
        return [self.registry.members[k]
                for k in sorted(self.registry.members,
                                key=lambda kk: self.registry.members[kk][
                                    "index"])]

    def slots(self) -> list[dict]:
        return [self.registry.slots[k]
                for k in sorted(self.registry.slots,
                                key=lambda kk: self.registry.slots[kk][
                                    "index"])]

    def slot_validity(self, origin: str, slot_key_name: str) -> dict | None:
        """Absolute UTC validity bounds of a slot for one origin."""

        return self.registry.slot_validity_map.get(origin, {}).get(
            slot_key_name)

    def source_of(self, origin: str) -> dict | None:
        if origin not in self._committed_set:
            return None
        return self.registry.sources.get(origin)

    def conflicts(self) -> list[dict]:
        return list(self.registry.conflicts)

    # ------------------------------------------------------------- selection
    def find_variable(self, variable: VariableId) -> str | None:
        return (variable_key(variable)
                if variable_key(variable) in self.registry.variables
                else None)

    def _filter_indices(self, entries: dict, selected_keys: set | None,
                        predicate) -> list[int]:
        ordered = sorted(entries.values(), key=lambda e: e["index"])
        out = []
        for entry in ordered:
            if selected_keys is not None and entry["key"] not in selected_keys:
                continue
            if predicate is not None and not predicate(entry):
                continue
            out.append(entry["index"])
        return out

    def select(
        self,
        variable: VariableId,
        *,
        levels: list[Level] | None = None,
        members: list[MemberId] | None = None,
        slots: list[TimeSlot] | None = None,
        origins: list[str] | None = None,
        validity_start_from: str | None = None,
        validity_end_to: str | None = None,
    ) -> SliceResult:
        """Return a stable, provenance-annotated slice of one variable.

        All filters are optional; an omitted filter means "every committed
        coordinate".  ``validity_start_from`` / ``validity_end_to`` apply to
        the absolute UTC validity boundaries stored per slot and use
        lexicographic comparison over the canonical ``...Z`` strings, which
        is chronological for identical formatting.
        """

        vk = self.find_variable(variable)
        if vk is None:
            raise KeyError(
                f"variable {variable_key(variable)} is not registered")
        var_entry = self.registry.variables[vk]

        level_keys = None if levels is None else {level_key(l) for l in levels}
        member_keys = None if members is None else {
            member_key(m) for m in members}
        slot_keys = None if slots is None else {slot_key(s) for s in slots}

        def slot_predicate_for(origin: str):
            """Validity window filter, evaluated per origin's absolute bounds."""

            def predicate(entry: dict) -> bool:
                if validity_start_from is None and validity_end_to is None:
                    return True
                bounds = self.slot_validity(origin, entry["key"])
                if bounds is None:
                    return False
                if validity_start_from is not None and \
                        bounds["validity_start"] < validity_start_from:
                    return False
                if validity_end_to is not None and \
                        bounds["validity_end"] > validity_end_to:
                    return False
                return True
            return predicate

        origin_indices = [
            i for i, o in enumerate(self.registry.origins)
            if o in self._committed_set and (
                origins is None or o in set(origins))]
        level_indices = self._filter_indices(
            self.registry.levels, level_keys, None)
        member_indices = self._filter_indices(
            self.registry.members, member_keys, None)
        base_slot_indices = self._filter_indices(
            self.registry.slots, slot_keys, None)

        # Validity windows are absolute and therefore origin-specific.  The
        # admissible slot set is computed per origin; when it is identical
        # across origins the result keeps the regular Cartesian layout, and
        # otherwise the differing per-origin sets are reported in the axes
        # (a common layout is retained whenever the filter is absent).
        per_origin_slot_indices: dict[int, list[int]] = {}
        for origin_idx in origin_indices:
            origin_name = self.registry.origins[origin_idx]
            if validity_start_from is None and validity_end_to is None:
                per_origin_slot_indices[origin_idx] = list(base_slot_indices)
            else:
                per_origin_slot_indices[origin_idx] = [
                    i for i in base_slot_indices
                    if slot_predicate_for(origin_name)(
                        self.registry.slots[self._key_at(
                            self.registry.slots, i)])]
        distinct_slot_sets = {
            tuple(v) for v in per_origin_slot_indices.values()}
        if len(distinct_slot_sets) != 1:
            raise KeyError(
                "validity window matches different slot sets per origin; "
                "query one origin at a time for per-origin windows")
        slot_indices = list(next(iter(distinct_slot_sets)))

        for label, idxs in (("levels", level_indices),
                            ("members", member_indices),
                            ("slots", slot_indices),
                            ("origins", origin_indices)):
            if not idxs:
                raise KeyError(f"selection matches no {label}")

        values_arr = self._group[f"vars/{vk}/values"]
        present_arr = self._group[f"vars/{vk}/present"]
        msg_arr = self._group[f"vars/{vk}/message_digest"]
        pkg_arr = self._group[f"vars/{vk}/package_digest"]

        oi = np.array(origin_indices, dtype="int64")
        li = np.array(level_indices, dtype="int64")
        mi = np.array(member_indices, dtype="int64")
        si = np.array(slot_indices, dtype="int64")

        # Orthogonal indexing pulls only the touched chunks and returns the
        # Cartesian product of the selected axes; nothing materialises the
        # full dataset in Python memory.
        selected_values = values_arr.oindex[
            oi, li, mi, si,
            np.arange(values_arr.shape[4]),
            np.arange(values_arr.shape[5])]
        selected_present = present_arr.oindex[oi, li, mi, si]
        selected_msg = msg_arr.oindex[oi, li, mi, si]
        selected_pkg = pkg_arr.oindex[oi, li, mi, si]

        axes = {
            "origin": [self.registry.origins[i] for i in origin_indices],
            "level": [self.registry.levels[self._key_at(
                self.registry.levels, i)]["identity"]
                for i in level_indices],
            "member": [self.registry.members[self._key_at(
                self.registry.members, i)]["identity"]
                for i in member_indices],
            "slot": [self._slot_axis_entry(i) for i in slot_indices],
            "latitude": [float(x) for x in self._group["coords/latitude"][:]],
            "longitude": [float(x) for x in self._group["coords/longitude"][:]],
        }
        return SliceResult(
            variable_identity=var_entry["identity"],
            units=var_entry["units"],
            axes=axes,
            values=np.asarray(selected_values),
            present=np.asarray(selected_present, dtype="uint8"),
            message_digest=np.asarray(selected_msg),
            package_digest=np.asarray(selected_pkg),
            selected={
                "variable_key": vk,
                "origin_indices": origin_indices,
                "level_indices": level_indices,
                "member_indices": member_indices,
                "slot_indices": slot_indices,
                "validity_start_from": validity_start_from,
                "validity_end_to": validity_end_to,
            })

    @staticmethod
    def _key_at(entries: dict, index: int) -> str:
        for key, entry in entries.items():
            if entry["index"] == index:
                return key
        raise KeyError(f"no registry entry with index {index}")

    def _slot_axis_entry(self, index: int) -> dict:
        """Slot label plus its per-origin absolute validity boundaries."""

        key = self._key_at(self.registry.slots, index)
        entry = dict(self.registry.slots[key])
        entry["validity_by_origin"] = {
            origin: self.slot_validity(origin, key)
            for origin in self._committed_origins
            if self.slot_validity(origin, key) is not None}
        return entry

    def committed_marker(self, origin: str) -> dict | None:
        path = (self.root / CUBE_DIR / "committed"
                / f"{marker_name(origin)}.json")
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
