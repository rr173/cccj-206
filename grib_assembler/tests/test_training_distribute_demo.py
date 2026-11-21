"""Fast acceptance test for the hand-drawn distribution demo script."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "distribution_demo.py"


def test_distribution_demo_chain_and_shards(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-I", str(SCRIPT), "--out", str(tmp_path / "demo")],
        cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    delivery = tmp_path / "demo" / "delivery"
    decisions = {
        (d["export_id"], d["sample_id"]): d
        for d in (json.loads(line) for line in
                  (delivery / "decisions.jsonl").read_text(
                      encoding="utf-8").splitlines())}
    # A and C are candidates pulled into a spanning two-window group.
    assert decisions[("export-2026-01", "A")]["decision"] == "isolated"
    assert decisions[("export-2026-01", "C")]["decision"] == "isolated"
    # D: exact-gap row stays separate and is delivered to validation.
    assert decisions[("export-2026-01", "D")]["decision"] == "validation"
    # H: independent holdout candidate is actually delivered.
    assert decisions[("export-2026-01", "H")]["decision"] == "holdout"
    # The explanation printed by the demo reaches C via message then obs.
    payload = proc.stdout
    assert '"edge_kind": "message"' in payload
    assert '"edge_kind": "observation"' in payload
    # The self-contained index exists and carries edges + references.
    assert (delivery / "distribution-index.sqlite").is_file()
    assert (delivery / "distribution-index.json").is_file()
