"""JSON-stable normalisation for structural identity comparisons.

Coordinate registries are persisted as JSON, where every tuple round-trips as
a list.  Compatibility checks must therefore compare *content* rather than
Python container types, while still rejecting any genuine semantic
difference.  Normalisation is recursive and limited to JSON-native types so
the rule applies uniformly to variables, levels, members and slots.
"""

from __future__ import annotations


def normalize(obj):
    """Return a JSON-canonical, content-based form of ``obj``.

    * tuples and lists both become tuples (order-sensitive);
    * dicts become lists of ``(key, normalize(value))`` sorted by key;
    * scalars (including ``None``, ints, floats, strings, bools) stay as-is.
    """

    if isinstance(obj, dict):
        return tuple(sorted(
            (k, normalize(v)) for k, v in obj.items()))
    if isinstance(obj, (list, tuple)):
        return tuple(normalize(v) for v in obj)
    return obj


def same_identity(left, right) -> bool:
    """True when two identity documents are semantically identical."""

    return normalize(left) == normalize(right)
