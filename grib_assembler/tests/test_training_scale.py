"""Pytest wrapper around the standalone scale/memory check.

Marked ``slow`` because it generates thousands of temporary samples; all
generated data stays under the system temp directory and is removed at the
end.  The check script is independent of the exporter's own data structures.
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
SCRIPT = ROOT / "tools" / "training_scale_check.py"


def test_scale_memory_and_single_value_block_reads():
    proc = subprocess.run(
        [sys.executable, "-I", str(SCRIPT),
         "--sizes", "200", "800", "2000",
         "--small-batch", "128", "--large-batch", "2048"],
        cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert report["sizes"] == [200, 800, 2000]
    for measurement in report["measurements"]:
        assert measurement["rows"] == measurement["samples"]
        assert measurement["value_block_reads"] == 1
        assert measurement["ok_cells"] == measurement["samples"]
    for slope in report["slopes"].values():
        assert slope < 5.0
    assert "at most" in report["value_block_rule"]
