"""Stable coordinate/identity keys for the analysis cube.

The cube is addressed along five analysis dimensions - reference time
(origin), level, ensemble member, time slot and variable - plus the spatial
grid.  Every key here is derived from the same numeric GRIB2 codes the
assembler core uses; ecCodes short names are never part of an identity.

The serialisations in this module are the *single* canonical forms stored in
Zarr attributes and sidecar evidence, so two imports of the same package on
different machines produce byte-identical metadata.
"""

from __future__ import annotations

import json

from ..identity import Level, MemberId, VariableId
from ..timeutil import TimeSlot, iso


def variable_key(var: VariableId) -> str:
    """Filesystem/attribute-safe canonical key of one variable identity.

    Two variables with identical short names but different numeric codes get
    different keys; the full identity is also stored as structured attributes.
    """

    doc = var.to_dict()
    return (
        "d{discipline}c{parameterCategory}p{parameterNumber}"
        "m{masterTablesVersionNumber}l{localTablesVersionNumber}"
        "ce{centre}s{subCentre}").format(**doc)


def variable_doc(var: VariableId) -> dict:
    return dict(var.to_dict())


def level_key(level: Level) -> str:
    first = level.first
    parts = [f"t{first.type}"]
    if first.value is None:
        parts.append("vNA")
    else:
        parts.append(
            f"v{first.value.numerator}d{first.value.denominator}")
    if level.second is None:
        parts.append("tNAvNA")
    else:
        second = level.second
        parts.append(f"t{second.type}")
        if second.value is None:
            parts.append("vNA")
        else:
            parts.append(
                f"v{second.value.numerator}d{second.value.denominator}")
    return "_".join(parts)


def level_doc(level: Level) -> dict:
    return level.to_dict()


def member_key(member: MemberId) -> str:
    if member.kind == "deterministic":
        return "deterministic"
    if member.kind == "control":
        return "control"
    if member.kind == "perturbed":
        return f"perturbed_{member.number}"
    # Unknown ensemble types keep their raw code rather than being coerced.
    return f"ensemble_type_{member.type_of_ensemble}_{member.number}"


def member_doc(member: MemberId) -> dict:
    return member.to_dict()


def slot_key(slot: TimeSlot) -> str:
    if slot.is_instant:
        return f"instant_{slot.start_seconds}"
    return f"stat{slot.statistic}_{slot.start_seconds}_{slot.end_seconds}"


def slot_doc(slot: TimeSlot) -> dict:
    return slot.to_dict()


def validity_bounds(reference_dt, slot: TimeSlot) -> tuple[str, str]:
    """Absolute UTC validity start/end datetimes for a slot."""

    from datetime import timedelta

    start = reference_dt + timedelta(seconds=slot.start_seconds)
    end = reference_dt + timedelta(seconds=slot.end_seconds)
    return iso(start), iso(end)


def canonical_json(obj: dict) -> str:
    """Deterministic JSON used for evidence/identity comparisons."""

    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)
