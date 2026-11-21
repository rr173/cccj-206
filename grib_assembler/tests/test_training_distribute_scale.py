"""Pytest wrapper around the standalone distribution scale/memory check.

Marked ``slow`` because it materializes thousands of temp rows; all generated
data stays under the system temp directory and is removed at the end.  The
check script is independent of the product grouping code's data structures:
its expected components are derived from the hand-authored fixture semantics.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "distribution_scale_check.py"


def test_distribution_scale_conservation_and_linear_edges():
    proc = subprocess.run(
        [sys.executable, "-I", str(SCRIPT),
         "--sizes", "200", "1000", "3000"],
        cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert report["sizes"] == [200, 1000, 3000]
    for measurement in report["measurements"]:
        scale = measurement["relation_scale"]
        assert scale["message_edges"] <= scale["successful_feature_refs"]
        assert scale["observation_edges"] < scale["rows"]
        assert scale["total_edges"] <= (
            scale["rows"] + scale["successful_feature_refs"])
        assert scale["forest_edges"] <= scale["rows"] - 1
        # Conservation: shard rows plus isolated rows equal every input row.
        total_delivered = (
            sum(measurement["shard_counts"].values())
            + measurement["isolated_count"])
        assert total_delivered == measurement["rows"]
    for slope in report["slopes"].values():
        assert slope < 5.0
