"""Typed errors for the offline training-data distribution stage.

A failure here always means *the distribution did not happen*: no shard,
isolation, decision or index file is published and every input export is left
untouched.  Errors carry a stable machine-readable ``reason`` plus an
``evidence`` object naming the offending manifest entry, row or feature so the
data administrator can locate and fix the input.  Ordinary business outcomes
(a row with no usable label, an interval outside every window, a connected
group whose candidates span two windows, ...) are never errors: they become
explicit, explained isolation records instead.
"""

from __future__ import annotations


class DistributionError(Exception):
    """Base error for distribution; carries structured evidence."""

    def __init__(self, reason: str, evidence: dict | None = None,
                 message: str | None = None):
        self.reason = reason
        self.evidence = evidence or {}
        super().__init__(message or reason)


class ManifestError(DistributionError):
    """The export manifest is malformed or internally inconsistent.

    Stable reasons include ``manifest_invalid``, ``manifest_file_missing``,
    ``manifest_export_id_missing``, ``manifest_export_id_not_stable``,
    ``manifest_export_path_not_unique`` and ``duplicate_export_id``.
    """


class PlanError(DistributionError):
    """The delivery plan is malformed or its windows overlap.

    Stable reasons include ``plan_invalid``, ``plan_window_invalid``,
    ``plan_windows_overlap``, ``plan_window_missing`` and
    ``gap_seconds_invalid``.
    """


class RowError(DistributionError):
    """One exported row is malformed or contradicts the distribution rules.

    Stable reasons include ``row_record_invalid``, ``row_identity_invalid``,
    ``duplicate_joint_identity``, ``row_interval_invalid`` and
    ``successful_feature_without_message_digest``.
    """
