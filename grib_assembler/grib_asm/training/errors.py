"""Typed errors for the point-in-time training export.

A failure here always means *the export did not happen*: no success JSONL or
summary file is left behind and neither the cube nor any input stream is
modified.  Every error carries a stable machine-readable ``reason`` and an
``evidence`` object naming the conflicting input and business key so a caller
can locate and fix the offending record.

Two broad families exist:

* :class:`TrainingExportError` - malformed/contradictory *input evidence*
  (a bad recipe, a same-second conflicting stream record, a map onto the wrong
  grid, a publish digest the cube cannot confirm, ...).  These are ordinary
  non-zero exits with a structured error document, never a silent missing
  value.
* ordinary value gaps (not registered, withdrawn, field absent, point
  missing, label retracted, ...) are *not* errors: they become explainable
  per-cell statuses in the output row.
"""

from __future__ import annotations


class TrainingExportError(Exception):
    """Base error for the training export; carries structured evidence."""

    def __init__(self, reason: str, evidence: dict | None = None,
                 message: str | None = None):
        self.reason = reason
        self.evidence = evidence or {}
        super().__init__(message or reason)


class RecipeError(TrainingExportError):
    """The recipe is internally inconsistent or contradicts the sample list.

    Stable reasons include ``recipe_invalid``, ``recipe_origin_mismatch``,
    ``label_cutoff_too_early`` and ``decision_after_target_start``.
    """


class StreamError(TrainingExportError):
    """A JSONL stream is malformed or self-contradictory.

    Stable reasons include ``stream_record_invalid``,
    ``stream_visibility_before_reference``, ``stream_duplicate_conflict`` and
    ``stream_ordering_violation``.
    """


class CubeEvidenceError(TrainingExportError):
    """Stream evidence contradicts the committed cube, or points are illegal.

    Stable reasons include ``station_grid_mismatch``,
    ``station_index_out_of_bounds``, ``publish_origin_not_committed``,
    ``publish_digest_mismatch``, ``field_variable_missing`` (only raised when a
    publish claims a cell the cube cannot represent - a genuinely absent
    optional cell is the ``field_missing`` status instead),
    ``cell_provenance_conflict`` and ``required_field_missing``.
    """
