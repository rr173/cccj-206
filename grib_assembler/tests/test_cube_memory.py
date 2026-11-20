"""Pytest wrapper around the independent streaming/memory script.

Marked ``slow`` because it generates many temporary cycles; all generated
data stays under the system temp directory and is removed at the end.  The
script itself is independent of the cube test helpers' expected-array
builders.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from e2e_helpers import HAS_ECCODES

pytestmark = [
    pytest.mark.skipif(not HAS_ECCODES,
                       reason="ecCodes required to encode fixtures"),
    pytest.mark.slow,
]

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "cube_stream_check.py"


def test_streaming_memory_scales_with_batch_not_cycles(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-I", str(SCRIPT),
         "--cycles", "24", "--grid", "10x8",
         "--small-batch", "1", "--large-batch", "8"],
        cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert report["cycles"] == 24
    # Every per-added-cycle slope is below the fixed flat-memory budget.
    assert report["slopes"]
    for label, slope in report["slopes"].items():
        assert slope < 0.25, (label, slope)
    assert "batch size" in report["conclusion"]
    # Both batch configurations were measured at both cycle counts.
    assert set(report["measurements"]) == {
        "batch1_small", "batch1_large",
        "batch8_small", "batch8_large"}
