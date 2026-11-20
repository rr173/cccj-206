"""Shared pytest fixtures and path setup."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture(scope="module")
def grid_digest_value():
    """Canonical grid digest of the standard demo grid (requires ecCodes)."""
    from demo_common import grid_digest_of
    dig, _ = grid_digest_of()
    return dig
