"""Point-in-time training-table export from a frozen analysis cube.

This sub-package is a *data preparation* stage.  It never trains a model,
never produces a forecast, never interpolates and never converts units.  Its
one job is to rebuild, for every sample in an input manifest, the feature row
that was genuinely visible at that sample's historical ``decision_at`` and to
attach a label resolved independently at a common ``label_cutoff``.

Two clocks are kept strictly separate:

* **feature clock** = the per-sample ``decision_at``.  Forecast
  publish/withdraw records and station map/unmap records participate only when
  their UTC visibility time is ``<= decision_at`` (equality counts as
  visible).  Observation visibility times never influence feature selection.
* **label clock** = the recipe-wide ``label_cutoff``.  Observation
  set/retract records for a sample's exact measurement key participate only
  when visible ``<= label_cutoff``; a later cutoff can therefore change a
  label without changing any feature.

The module layout mirrors the staged pipeline:

* :mod:`errors`      typed, evidence-bearing failure reasons;
* :mod:`model`       recipe/sample/stream record dataclasses and validation;
* :mod:`timeline`    per-business-key visible-record resolution (two clocks);
* :mod:`workspace`   disk-backed (SQLite) staging so the inputs never have to
                     fit in memory;
* :mod:`cubeaccess`  bounded, chunk-once point reads against the Zarr cube;
* :mod:`export`      the orchestrating pipeline and the JSONL/JSON writer;
* :mod:`cli`         the ``cube-training-export`` command.
"""

from .errors import TrainingExportError
from .export import ExportConfig, TrainingExporter, export_training_table
from .model import FeatureColumn, Recipe, Sample

__all__ = [
    "TrainingExportError",
    "ExportConfig",
    "TrainingExporter",
    "export_training_table",
    "FeatureColumn",
    "Recipe",
    "Sample",
]
