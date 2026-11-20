"""Typed errors for the package-to-cube pipeline.

Every error carries a stable machine-readable ``reason`` code so that failure
reports stay identical across runs and builds; free-form details live in
``evidence``.  Any failure raised here means *nothing* was committed to the
published dataset.
"""

from __future__ import annotations


class CubeError(Exception):
    """Base error for cube pipeline failures."""

    def __init__(self, reason: str, evidence: dict | None = None,
                 message: str | None = None):
        self.reason = reason
        self.evidence = evidence or {}
        super().__init__(message or reason)


class PackageValidationError(CubeError):
    """The hand-off package failed an independent integrity/identity check."""


class UnsupportedCubeMessage(CubeError):
    """A well-formed, verified message cannot be represented in the cube.

    Typical stable reasons:

    * ``unsupported_grid_template``  - not a regular lat/lon (GDT 3.0) grid;
    * ``unsupported_scanning_mode``  - scan order other than row-major (0);
    * ``undecodable_message``        - ecCodes cannot fully decode the field;
    * ``point_count_mismatch``       - decoded point count != Ni * Nj;
    * ``grid_coordinate_mismatch``   - computed coordinates disagree with
      the coordinates ecCodes decodes from the same message;
    * ``unsupported_pdt`` / the profile-level codes from
      :mod:`grib_asm.classify`, surfaced verbatim when a record was marked
      unsupported inside the package itself.
    """


class CubeCompatibilityError(CubeError):
    """A new package is incompatible with an already published dataset.

    Stable reasons: ``grid_definition_conflict``,
    ``variable_identity_conflict``, ``level_semantics_conflict``,
    ``unit_conflict``, ``cf_mapping_conflict``.
    """


class CubeConflictError(CubeError):
    """Same reference time, different whole-package digest.

    The old values stay in place and readable; the evidence of the second
    source is retained under ``conflicts/`` and recorded in the index.
    """
