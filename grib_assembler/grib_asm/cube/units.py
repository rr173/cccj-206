"""Conservative CF standard-name / unit mapping.

The pipeline never guesses.  A mapping is applied only when an explicit,
versioned rule matches the *full* numeric variable identity
(discipline, category, parameter number, master/local table versions,
centre/sub-centre).  Everything else keeps its raw GRIB unit and carries no
CF name, and the stored ``mapping_status`` says exactly why.

The table is intentionally tiny: it exists to prove the mechanism and the
conflict rules, not to claim coverage of the WMO registry.  Adding an entry
is a reviewable code change; the mapping can never be inferred from a short
name at import time.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..identity import VariableId

# Keyed by the full numeric identity.  ``None`` means "no credible CF mapping";
# the raw unit is still retained verbatim.
CF_TABLE: dict[tuple, tuple[str | None, str | None]] = {
    # WMO master table 29, discipline 0 (meteorological), category 0
    # (temperature), parameter 0 = temperature, centre 7 (ECMWF), sub 0.
    (0, 0, 0, 29, 1, 7, 0): ("air_temperature", "K"),
    # category 1 (moisture), parameter 1 = specific humidity.
    (0, 1, 1, 29, 1, 7, 0): ("specific_humidity", "kg kg**-1"),
    # category 3 (mass), parameter 0 = pressure.
    (0, 3, 0, 29, 1, 7, 0): ("air_pressure", "Pa"),
}

MAPPED = "mapped"
RAW_ONLY = "raw_only_no_credible_mapping"


@dataclass(frozen=True)
class UnitInfo:
    raw_unit: str
    cf_standard_name: str | None
    cf_canonical_unit: str | None
    mapping_status: str
    mapping_source: str

    def doc(self) -> dict:
        return {
            "raw_unit": self.raw_unit,
            "cf_standard_name": self.cf_standard_name,
            "cf_canonical_unit": self.cf_canonical_unit,
            "mapping_status": self.mapping_status,
            "mapping_source": self.mapping_source,
        }


def resolve_units(var: VariableId, raw_unit: str) -> UnitInfo:
    key = (var.discipline, var.parameter_category, var.parameter_number,
           var.master_tables_version, var.local_tables_version,
           var.centre, var.sub_centre)
    entry = CF_TABLE.get(key)
    if entry is None:
        return UnitInfo(
            raw_unit=raw_unit, cf_standard_name=None,
            cf_canonical_unit=None, mapping_status=RAW_ONLY,
            mapping_source="none")
    cf_name, cf_unit = entry
    return UnitInfo(
        raw_unit=raw_unit, cf_standard_name=cf_name, cf_canonical_unit=cf_unit,
        mapping_status=MAPPED,
        mapping_source="grib_asm.cube.units.CF_TABLE@v1")
