"""Canonical identities: variable, level, ensemble member, grid, business key.

Identities are deliberately built from raw GRIB2 numeric codes rather than
ecCodes short names.  Two messages with the same short name but different
parameter numbers are different variables; a surface and a 0 Pa pressure level
are different levels; an undefined second fixed surface stays ``missing`` and
is never coerced to zero.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction

from .timeutil import TimeSlot

# GRIB2 "missing" sentinel for byte / short / long scale factors and values.
MISSING_BYTE = 255
MISSING_LONG = 0xFFFFFFFF


def scaled_value(scale_factor: int, scaled_value: int) -> Fraction | None:
    """Turn (scale factor, scaled value) into an exact rational level value.

    Value = scaledValue / 10**scaleFactor.  A missing scale factor or value
    means the surface is undefined; we return ``None`` instead of inventing 0.
    """

    if scale_factor == MISSING_BYTE or scaled_value == MISSING_LONG:
        return None
    # ecCodes may hand back a negative scale factor (GRIB2 octet stores an
    # already-decoded signed byte).  Keep everything in integer arithmetic so
    # the result stays an exact rational and never becomes a float.
    if scale_factor >= 0:
        return Fraction(scaled_value, 10 ** scale_factor)
    return Fraction(scaled_value * 10 ** (-scale_factor), 1)


@dataclass(frozen=True)
class FixedSurface:
    """One fixed surface, normalised to an exact rational value.

    The raw GRIB2 ``scale_factor`` and integer ``scaled_value`` are retained so
    the audit record round-trips to the original octets; ``value`` is the exact
    rational ``scaled_value / 10**scale_factor`` and is ``None`` only when the
    surface is genuinely missing.
    """

    type: int
    value: Fraction | None  # None means the surface value is genuinely missing
    scale_factor: int | None = None
    scaled_value: int | None = None

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "scale_factor": self.scale_factor,
            "scaled_value": self.scaled_value,
            "rational": None if self.value is None else (
                self.value.numerator, self.value.denominator),
        }


def make_surface(stype: int, factor: int, value: int) -> FixedSurface:
    rational = scaled_value(factor, value)
    if rational is None:
        # Genuinely undefined surface: keep no factor/value, never zero.
        return FixedSurface(type=int(stype), value=None,
                            scale_factor=None, scaled_value=None)
    return FixedSurface(type=int(stype), value=rational,
                        scale_factor=int(factor), scaled_value=int(value))


@dataclass(frozen=True)
class Level:
    """A level is the ordered pair of fixed surfaces (first, second)."""

    first: FixedSurface
    second: FixedSurface | None  # None = no second surface defined

    def to_dict(self) -> dict:
        return {
            "first": self.first.to_dict(),
            "second": None if self.second is None else self.second.to_dict(),
        }


@dataclass(frozen=True)
class VariableId:
    discipline: int
    parameter_category: int
    parameter_number: int
    master_tables_version: int
    local_tables_version: int
    centre: int
    sub_centre: int

    def to_dict(self) -> dict:
        return {
            "discipline": self.discipline,
            "parameterCategory": self.parameter_category,
            "parameterNumber": self.parameter_number,
            "masterTablesVersionNumber": self.master_tables_version,
            "localTablesVersionNumber": self.local_tables_version,
            "centre": self.centre,
            "subCentre": self.sub_centre,
        }


# GRIB2 Table 4.6 - Type of ensemble forecast.
PERTURBED_CATEGORY = 1  # Ensemble forecast (perturbed)
CONTROL_CATEGORY = 3    # Control forecast


@dataclass(frozen=True)
class MemberId:
    """Ensemble membership.

    Deterministic products carry ``kind="deterministic"``.  Ensemble products
    carry kind ``"control"`` (type 3) or ``"perturbed"`` (type 1) plus the
    perturbation number.
    """

    kind: str
    type_of_ensemble: int | None = None
    number: int | None = None

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "typeOfEnsembleForecast": self.type_of_ensemble,
            "perturbationNumber": self.number,
        }


def deterministic_member() -> MemberId:
    return MemberId(kind="deterministic", type_of_ensemble=None, number=None)


def ensemble_member(e_type: int, number: int) -> MemberId:
    if e_type == CONTROL_CATEGORY:
        kind = "control"
    elif e_type == PERTURBED_CATEGORY:
        kind = "perturbed"
    else:
        # Keep the raw code but label it generically; it still has a stable
        # identity and never silently becomes a control/deterministic member.
        kind = f"ensemble_type_{e_type}"
    return MemberId(kind=kind, type_of_ensemble=int(e_type),
                    number=int(number))


def grid_digest(gds_bytes: bytes) -> str:
    """Identity of a grid from the *complete* Section 3 bytes.

    Packing lives in Sections 5/7 and is therefore excluded: changing the
    packing does not change the grid identity.
    """

    return hashlib.sha256(b"GRIB2-GDS-v1\x00" + gds_bytes).hexdigest()


@dataclass(frozen=True)
class BusinessKey:
    """Canonical key of one request cell."""

    reference_time: str       # ISO UTC
    grid_digest: str
    variable: VariableId
    level: Level
    member: MemberId
    slot: TimeSlot

    def canonical(self) -> str:
        payload = {
            "reference_time": self.reference_time,
            "grid_digest": self.grid_digest,
            "variable": self.variable.to_dict(),
            "level": self.level.to_dict(),
            "member": self.member.to_dict(),
            "slot": self.slot.to_dict(),
        }
        # Sort keys so the serialization is independent of insertion order.
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False)
        return blob

    def digest(self) -> str:
        return hashlib.sha256(
            b"GRIB2-BKEY-v1\x00" + self.canonical().encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        return json.loads(self.canonical())
