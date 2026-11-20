"""Per-cycle readiness and whole-request cycle selection.

For every candidate reference time inside the requested origin range and not
later than ``not_after``, the request Cartesian product (grid x variable x
level x member x slot) is checked against catalog cells.  A cycle is complete
only when *every* cell exists, matches the expected grid and is ``filled``
(not conflicted).  Selection never mixes cycles: it picks the newest complete
cycle, or reports ``blocked`` with explicit reasons.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from .catalog import Catalog
from .identity import BusinessKey
from .request import Request
from .timeutil import iso


@dataclass
class CellStatus:
    index: int
    reference_time: str
    grid_digest: str
    variable: dict
    level: dict
    member: dict
    slot: dict
    bkey_digest: str
    state: str                 # filled | conflict | missing | grid_mismatch
    detail: dict = field(default_factory=dict)


@dataclass
class CycleReadiness:
    reference_time: str
    total: int
    filled: int
    missing: list[CellStatus]
    conflicts: list[CellStatus]
    grid_mismatches: list[CellStatus]
    complete: bool

    def reasons(self) -> list[dict]:
        out = []
        for cs in self.missing:
            out.append({"reason": "missing", "cell": _brief(cs)})
        for cs in self.conflicts:
            out.append({"reason": "conflict", "cell": _brief(cs),
                        "detail": cs.detail})
        for cs in self.grid_mismatches:
            out.append({"reason": "grid_mismatch", "cell": _brief(cs),
                        "detail": cs.detail})
        return out


def _brief(cs: CellStatus) -> dict:
    return {
        "index": cs.index,
        "grid_digest": cs.grid_digest,
        "variable": cs.variable,
        "level": cs.level,
        "member": cs.member,
        "slot": cs.slot,
        "bkey_digest": cs.bkey_digest,
    }


def iter_request_keys(req: Request, reference_time: str):
    """Yield (index, BusinessKey) over the request Cartesian product."""

    idx = 0
    for var in req.variables:
        for lvl in req.levels:
            for mem in req.members:
                for slot in req.slots:
                    key = BusinessKey(
                        reference_time=reference_time,
                        grid_digest=req.expected_grid_digest,
                        variable=var.as_variable_id(),
                        level=lvl.as_level(),
                        member=mem.as_member_id(),
                        slot=slot,
                    )
                    yield idx, key
                    idx += 1


def evaluate_cycle(catalog: Catalog, req: Request,
                   reference_time: str) -> CycleReadiness:
    missing: list[CellStatus] = []
    conflicts: list[CellStatus] = []
    grid_mismatches: list[CellStatus] = []
    filled = 0
    total = 0

    for idx, key in iter_request_keys(req, reference_time):
        total += 1
        digest = key.digest()
        cell = catalog.cell(digest)
        cs = CellStatus(
            index=idx,
            reference_time=reference_time,
            grid_digest=req.expected_grid_digest,
            variable=key.variable.to_dict(),
            level=key.level.to_dict(),
            member=key.member.to_dict(),
            slot=key.slot.to_dict(),
            bkey_digest=digest,
            state="missing",
        )
        if cell is None:
            # A message for another grid can never substitute.  Look for any
            # cell sharing every dimension except the grid digest.
            near = _find_other_grid(catalog, key)
            if near is not None:
                cs.state = "grid_mismatch"
                cs.detail = near
                grid_mismatches.append(cs)
            else:
                missing.append(cs)
            continue
        if cell["state"] == "conflict":
            cs.state = "conflict"
            cs.detail = {
                "distinct_message_digests": cell["distinct_digests"],
                "sources": cell["sources"],
            }
            conflicts.append(cs)
            continue
        filled += 1
        cs.state = "filled"

    return CycleReadiness(
        reference_time=reference_time,
        total=total,
        filled=filled,
        missing=missing,
        conflicts=conflicts,
        grid_mismatches=grid_mismatches,
        complete=(filled == total),
    )


def _find_other_grid(catalog: Catalog, key: BusinessKey) -> dict | None:
    """Detect a same-business-key message that only differs by grid.

    Scans catalog messages of this reference time and compares every identity
    field except ``grid_digest``.  This is evidence only; it cannot fill the
    requested cell.
    """

    rows = catalog._conn.execute(
        "SELECT identity, grid_digest, message_sha256, collection_id, file_id "
        "FROM messages WHERE reference_time=? AND status='ok'",
        (key.reference_time,)).fetchall()
    target = key.to_dict()
    target_var = target["variable"]
    target_lvl = target["level"]
    target_mem = target["member"]
    target_slot = target["slot"]
    for row in rows:
        ident = json.loads(row["identity"])
        if ident["grid_digest"] == key.grid_digest:
            continue
        if (ident["variable"] == target_var
                and ident["level"] == target_lvl
                and ident["member"] == target_mem
                and ident["slot"] == target_slot):
            return {
                "other_grid_digest": row["grid_digest"],
                "message_sha256": row["message_sha256"],
                "collection_id": row["collection_id"],
                "file_id": row["file_id"],
            }
    return None


def candidate_cycles(req: Request,
                     known_reference_times: list[str]) -> list[datetime]:
    lo = req.origin_from
    hi = min(req.origin_to, req.not_after)
    out = []
    for text in known_reference_times:
        dt = datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=lo.tzinfo)
        if lo <= dt <= hi:
            out.append(dt)
    out.sort()
    return out


@dataclass
class ReadinessReport:
    request_id: str
    not_after: str
    expected_cells: int
    cycles: list[CycleReadiness]
    selected_reference_time: str | None
    blocked: bool
    selection_reason: str

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "not_after": self.not_after,
            "expected_cells": self.expected_cells,
            "selected_reference_time": self.selected_reference_time,
            "blocked": self.blocked,
            "selection_reason": self.selection_reason,
            "cycles": [
                {
                    "reference_time": c.reference_time,
                    "total": c.total,
                    "filled": c.filled,
                    "complete": c.complete,
                    "reasons": c.reasons(),
                } for c in self.cycles
            ],
        }


def build_readiness(catalog: Catalog, req: Request) -> ReadinessReport:
    known = catalog.cycle_reference_times()
    cycles_dt = candidate_cycles(req, known)
    readiness = [evaluate_cycle(catalog, req, iso(dt)) for dt in cycles_dt]

    selected = None
    reason = ""
    # Newest complete cycle wins; walk newest -> oldest.
    for cyc in reversed(readiness):
        if cyc.complete:
            selected = cyc.reference_time
            reason = "newest complete cycle at or before not_after"
            break
    blocked = selected is None
    if blocked:
        reason = ("no complete cycle in origin range at or before not_after; "
                  "see per-cycle missing/conflict/grid_mismatch reasons")
    return ReadinessReport(
        request_id=req.request_id,
        not_after=iso(req.not_after),
        expected_cells=req.expected_cells(),
        cycles=readiness,
        selected_reference_time=selected,
        blocked=blocked,
        selection_reason=reason,
    )
