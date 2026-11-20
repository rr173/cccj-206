"""Explain the resolution of a single request slot/cell.

``explain-slot`` runs independently of ``assemble``: given a request, a
reference time and one full cell coordinate, it reports whether a usable
message exists, every retransmission source, any conflict (with the distinct
message digests), and near-miss evidence such as another grid, another level
or another time range that must not substitute for the cell.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime

from .catalog import Catalog
from .identity import BusinessKey
from .request import Request, RequestedLevel, RequestedMember, RequestedVariable
from .timeutil import TimeSlot, iso, slot_from_dict


@dataclass
class SlotExplanation:
    reference_time: str
    business_key_digest: str
    state: str
    cell: dict | None
    near_misses: list[dict]
    narrative: str

    def to_dict(self) -> dict:
        return {
            "reference_time": self.reference_time,
            "business_key_digest": self.business_key_digest,
            "state": self.state,
            "cell": self.cell,
            "near_misses": self.near_misses,
            "narrative": self.narrative,
        }


def build_key(req: Request, reference_time: str, variable_index: int,
              level_index: int, member_index: int,
              slot_index: int) -> BusinessKey:
    var = req.variables[variable_index]
    lvl = req.levels[level_index]
    mem = req.members[member_index]
    slot = req.slots[slot_index]
    return BusinessKey(
        reference_time=reference_time,
        grid_digest=req.expected_grid_digest,
        variable=var.as_variable_id(),
        level=lvl.as_level(),
        member=mem.as_member_id(),
        slot=slot,
    )


def explain_slot(catalog: Catalog, req: Request, reference_time: str,
                 variable_index: int, level_index: int,
                 member_index: int, slot_index: int) -> SlotExplanation:
    key = build_key(req, reference_time, variable_index, level_index,
                    member_index, slot_index)
    digest = key.digest()
    cell = catalog.cell(digest)
    near = _near_misses(catalog, key)

    if cell is None:
        state = "missing"
        if near:
            narrative = (
                "no message matches the full canonical business key; near-miss "
                "messages exist but differ on a non-substitutable dimension "
                "(grid/level/slot/variable/member) and cannot fill this cell")
        else:
            narrative = "no message anywhere matches this business key"
    elif cell["state"] == "conflict":
        state = "conflict"
        narrative = (
            "the cell has multiple messages with distinct whole-message "
            "SHA-256 digests; it is unusable regardless of file recency")
    else:
        state = "filled"
        nsrc = len(cell["sources"])
        narrative = (
            f"usable cell; {nsrc} source location(s), all sharing one message "
            "digest (retransmissions occupy this single cell)")

    return SlotExplanation(
        reference_time=reference_time,
        business_key_digest=digest,
        state=state,
        cell=cell,
        near_misses=near,
        narrative=narrative,
    )


def _near_misses(catalog: Catalog, key: BusinessKey) -> list[dict]:
    rows = catalog._conn.execute(
        "SELECT identity, grid_digest, message_sha256, collection_id, "
        "file_id, msg_index, reference_time FROM messages "
        "WHERE status='ok' AND reference_time=?",
        (key.reference_time,)).fetchall()
    target = key.to_dict()
    out: list[dict] = []
    for row in rows:
        ident = json.loads(row["identity"])
        if ident["grid_digest"] == key.grid_digest \
                and ident["variable"] == target["variable"] \
                and ident["level"] == target["level"] \
                and ident["member"] == target["member"] \
                and ident["slot"] == target["slot"]:
            continue  # exact identity handled by cell lookup
        # Keep only messages sharing at least variable + member + reference so
        # the report explains a concrete near-miss rather than the whole archive.
        if (ident["variable"] == target["variable"]
                and ident["member"] == target["member"]):
            diffs = []
            if ident["grid_digest"] != key.grid_digest:
                diffs.append("grid")
            if ident["level"] != target["level"]:
                diffs.append("level")
            if ident["slot"] != target["slot"]:
                diffs.append("slot")
            out.append({
                "differ_on": diffs,
                "other_grid_digest": row["grid_digest"],
                "other_identity": ident,
                "collection_id": row["collection_id"],
                "file_id": row["file_id"],
                "msg_index": row["msg_index"],
                "message_sha256": row["message_sha256"],
            })
    return out
