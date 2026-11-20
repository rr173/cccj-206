"""Offline "hand-off package -> analysis cube" pipeline.

This sub-package turns *verified* GRIB2 hand-off packages (see
:mod:`grib_asm.verify`) into an appendable, slice-readable Zarr analysis
dataset.  In contrast to the byte-preserving assembler core, this is the one
place grid point values are decoded through ecCodes; it still never
re-selects messages, interpolates, corrects, computes ensemble statistics or
produces any forecast.

Hard rules enforced here:

* a package is imported only after the completion marker, the whole data-file
  digest, the per-message offset/length/digest, message order and the original
  request identity have all been independently verified;
* only regular latitude/longitude grids whose single message decodes
  completely are accepted; every other grid or undecodable message gets a
  stable, machine-readable ``unsupported_*`` reason and the cycle is *not*
  reported complete;
* variables keep their full numeric identity (discipline, parameter
  category/number, table versions, centre/sub-centre); short names never merge
  variables;
* instantaneous fields and statistical intervals stay distinguishable via
  explicit start/end seconds relative to the reference time, the statistical
  process code and absolute validity boundaries;
* missing cells are represented by an explicit mask, never by values copied
  from a neighbouring cycle;
* imports commit per cycle: a crash can only leave previously committed cycles
  visible, never a half-registered one;
* memory is bounded by the configured batch size and the single-message grid
  size, never by the total number of cycles.
"""

from .errors import (
    CubeError, PackageValidationError, UnsupportedCubeMessage,
    CubeConflictError, CubeCompatibilityError,
)
from .importer import CubeConfig, ImportReport, import_package
from .query import CubeReader, SliceResult
from .recovery import recover

__all__ = [
    "CubeError",
    "PackageValidationError",
    "UnsupportedCubeMessage",
    "CubeConflictError",
    "CubeCompatibilityError",
    "CubeConfig",
    "ImportReport",
    "import_package",
    "CubeReader",
    "SliceResult",
    "recover",
]
