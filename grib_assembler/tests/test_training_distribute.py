"""End-to-end tests for offline connected-source distribution.

The expected outcomes here are computed by ``NaiveJudge``, a deliberately
independent reimplementation that enumerates *every* related pair directly:

* message relations: every pair of rows that both carry one identical
  ``message_digest`` in an ``ok`` feature;
* observation relations: every pair of same ``(station_id, metric)`` rows whose
  gap-extended half-open intervals intersect with positive length.

It then computes transitive closure itself and applies the delivery rules.
None of the product grouping code (DSU, sweep, edge builders) is used to
derive an expectation - the product code is only checked against the judge.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import pytest

from grib_asm.training.distribute import (
    DistributionConfig, run_distribution,
)
from grib_asm.training.distribute_errors import DistributionError
from grib_asm.training.distribute_explain import explain_row
from grib_asm.training.distribute_rows import joint_identity
from grib_asm.training.distribute_workspace import (
    DistributionWorkspace,
)

ROOT = Path(__file__).resolve().parents[1]
H = 3600
# 2026-01-01T00:00:00Z
T0 = 1767225600


# --------------------------------------------------------------------------- #
# row authoring
# --------------------------------------------------------------------------- #
def digest(letter: str) -> str:
    # Real 64-lowercase-hex shape; equal letters yield equal digests.
    return hashlib.sha256(letter.encode("utf-8")).hexdigest()


def make_row(sample_id, station, metric, start_h, end_h, *,
             label="ok", digests=("M",), columns=None):
    if columns is None:
        columns = [f"col{i}" for i in range(len(digests))]
    return {
        "sample_id": sample_id,
        "station_id": station,
        "metric": metric,
        "target_interval": [T0 + int(start_h * H), T0 + int(end_h * H)],
        "decision_at": "2026-01-01T00:00:00Z",
        "features": [
            {"column": col, "status": "ok", "value": 1.5,
             "evidence": {"field": {"message_digest": digest(letter),
                                    "package_digest": "pkg-ignored"}}}
            for col, letter in zip(columns, digests)],
        "label": {"status": label,
                  "value": 1.0 if label == "ok" else None, "unit": "K"},
    }


def default_plan(gap_seconds=H):
    return {
        "gap_seconds": gap_seconds,
        "windows": {
            "train": {"start": "2026-01-01T01:00:00Z",
                      "end": "2026-01-01T04:00:00Z"},
            "validation": {"start": "2026-01-01T04:00:00Z",
                           "end": "2026-01-01T10:00:00Z"},
            "holdout": {"start": "2026-01-01T12:00:00Z",
                        "end": "2026-01-01T18:00:00Z"}}}


def write_run(tmp_path, exports, plan, *, config=None):
    """Write exports + manifest + plan, run distribution, return paths."""

    manifest = []
    for export_id, rows in exports:
        path = tmp_path / f"{export_id}.jsonl"
        path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8")
        manifest.append({"export_id": export_id, "path": str(path)})
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    out_dir = tmp_path / "out"
    report = run_distribution(manifest_path, plan_path, out_dir,
                              config=config)
    return manifest, plan_path, out_dir, report


def load_decisions(out_dir: Path):
    return {
        (d["export_id"], d["sample_id"]): d
        for d in (json.loads(line) for line in
                  (out_dir / "decisions.jsonl").read_text(
                      encoding="utf-8").splitlines())}


# --------------------------------------------------------------------------- #
# independent brute-force judge
# --------------------------------------------------------------------------- #
class NaiveJudge:
    """Enumerate all relations and derive every expected decision directly."""

    def __init__(self, exports, plan):
        self.gap = plan["gap_seconds"]
        self.windows = plan["windows"]
        self.rows = {}
        for export_id, rows in exports:
            for row in rows:
                jid = joint_identity(export_id, row["sample_id"])
                assert jid not in self.rows
                self.rows[jid] = row
        self.direct_reasons = {jid: [] for jid in self.rows}
        self.candidate_window = {}
        for jid, row in self.rows.items():
            if row["label"]["status"] != "ok":
                self.direct_reasons[jid].append("label_unavailable")
            relation = self._window_relation(row)
            if relation == "crosses_windows":
                self.direct_reasons[jid].append("crosses_windows")
            elif relation == "outside_windows":
                self.direct_reasons[jid].append("outside_windows")
            if row["label"]["status"] == "ok" and relation == "in_window":
                self.candidate_window[jid] = self._containing_window(row)
        self.components = self._components()

    def _window_relation(self, row):
        start, end = row["target_interval"]
        inside = [name for name, w in self.windows.items()
                  if self._epoch(w["start"]) <= start
                  and end <= self._epoch(w["end"])]
        if len(inside) == 1:
            return "in_window"
        if any(self._epoch(w["start"]) < end
               and start < self._epoch(w["end"])
               for w in self.windows.values()):
            return "crosses_windows"
        return "outside_windows"

    def _containing_window(self, row):
        start, end = row["target_interval"]
        for name, w in self.windows.items():
            if (self._epoch(w["start"]) <= start
                    and end <= self._epoch(w["end"])):
                return name
        raise AssertionError

    @staticmethod
    def _epoch(text):
        from datetime import datetime, timezone
        return int(datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")
                   .replace(tzinfo=timezone.utc).timestamp())

    def _message_related(self, r1, r2):
        digests1 = {
            f["evidence"]["field"]["message_digest"]
            for f in r1["features"] if f["status"] == "ok"}
        digests2 = {
            f["evidence"]["field"]["message_digest"]
            for f in r2["features"] if f["status"] == "ok"}
        return bool(digests1 & digests2)

    def _obs_related(self, r1, r2):
        if (r1["station_id"], r1["metric"]) != \
                (r2["station_id"], r2["metric"]):
            return False
        a0, a1 = r1["target_interval"]
        b0, b1 = r2["target_interval"]
        # provisional [start, end+gap); positive-length intersection only
        return a0 < b1 + self.gap and b0 < a1 + self.gap

    def _components(self):
        parent = {jid: jid for jid in self.rows}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for jid1, jid2 in combinations(sorted(self.rows), 2):
            r1, r2 = self.rows[jid1], self.rows[jid2]
            if self._message_related(r1, r2) or self._obs_related(r1, r2):
                parent[find(jid1)] = find(jid2)
        groups = {}
        for jid in self.rows:
            groups.setdefault(find(jid), set()).add(jid)
        # name each component by its smallest joint identity (UTF-8 order)
        return {min(members): members for members in groups.values()}

    def expected(self):
        """Return ``{joint_id: (decision, reasons, group_min)}``."""

        result = {}
        for group_min, members in self.components.items():
            windows = {self.candidate_window[j] for j in members
                       if j in self.candidate_window}
            for jid in members:
                reasons = list(self.direct_reasons[jid])
                if jid in self.candidate_window and len(windows) >= 2:
                    reasons.append("group_spans_multiple_windows")
                if jid in self.candidate_window and len(windows) == 1:
                    decision = next(iter(windows))
                else:
                    decision = "isolated"
                result[jid] = (decision, reasons, group_min)
        return result


# --------------------------------------------------------------------------- #
# the hand-drawn isolation chain scenario
# --------------------------------------------------------------------------- #
def hand_drawn_exports():
    # A (train candidate) --same message M1--> B (no usable label);
    # B --observation overlap--> C (validation candidate); A and C share no
    # direct relation. D shares C's station/metric with an exact-gap interval
    # and must stay disconnected. H is a fully independent holdout candidate.
    exp1 = [
        make_row("A", "ST1", "temp", 1, 3, digests=("A",)),
        make_row("B", "ST1", "temp", 3, 4, label="label_retracted",
                 digests=("A",), columns=("other-col",)),
        make_row("C", "ST1", "temp", 4, 6, digests=("C",)),
        make_row("D", "ST1", "temp", 7, 8, digests=("D",)),
        make_row("H", "ST9", "rain", 13, 14, digests=("H",)),
    ]
    # Cross-export same message: X (train window) and Y (holdout window)
    # reference the same digest from different packages; they must connect.
    exp2 = [
        make_row("X", "ST7", "temp", 1, 2, digests=("X",)),
        make_row("Y", "ST8", "temp", 15, 16, digests=("X",)),
        # E spans a window boundary; F lies outside every window.
        make_row("E", "ST5", "temp", 3, 12, digests=("E",)),
        make_row("F", "ST6", "temp", 20, 21, digests=("F",)),
    ]
    return [("exp-one", exp1), ("exp-two", exp2)]


def test_hand_drawn_chain_matches_naive_judge(tmp_path):
    exports = hand_drawn_exports()
    plan = default_plan()
    _, _, out_dir, report = write_run(tmp_path, exports, plan)
    decisions = load_decisions(out_dir)
    judge = NaiveJudge(exports, plan)
    expected = judge.expected()

    assert set(decisions) == {tuple(jid.split("\x1f", 1))
                              for jid in expected}
    for jid, (decision, reasons, group_min) in expected.items():
        key = tuple(jid.split("\x1f", 1))
        doc = decisions[key]
        assert doc["decision"] == decision, jid
        assert sorted(doc["reasons"]) == sorted(reasons), jid
        assert (doc["group"]["export_id"], doc["group"]["sample_id"]) == \
            tuple(group_min.split("\x1f", 1)), jid

    # Headline hand-drawn expectations.
    a = decisions[("exp-one", "A")]
    c = decisions[("exp-one", "C")]
    assert a["decision"] == "isolated"
    assert a["reasons"] == ["group_spans_multiple_windows"]
    assert c["decision"] == "isolated"
    # D: exact gap, disconnected, delivered as a validation candidate.
    d = decisions[("exp-one", "D")]
    assert d["decision"] == "validation" and d["reasons"] == []
    # H: fully independent holdout candidate.
    assert decisions[("exp-one", "H")]["decision"] == "holdout"
    # X and Y isolated together via a cross-export shared message.
    assert decisions[("exp-two", "X")]["decision"] == "isolated"
    assert decisions[("exp-two", "Y")]["decision"] == "isolated"
    assert decisions[("exp-two", "X")]["group"] == \
        decisions[("exp-two", "Y")]["group"]
    # Conservation and counts.
    assert report.shard_counts["train"] == 0
    assert report.shard_counts["validation"] == 1
    assert report.shard_counts["holdout"] == 1
    assert (report.shard_counts["train"]
            + report.shard_counts["validation"]
            + report.shard_counts["holdout"]
            + report.isolated_count) == report.row_count


def test_delivery_files_preserve_raw_rows_and_export_id(tmp_path):
    exports = hand_drawn_exports()
    _, _, out_dir, _ = write_run(tmp_path, exports, default_plan())
    seen = set()
    for name in ("train", "validation", "holdout", "isolated"):
        for line in (out_dir / f"{name}.jsonl").read_text(
                encoding="utf-8").splitlines():
            doc = json.loads(line)
            assert set(doc) == {"export_id", "row"}
            jid = (doc["export_id"], doc["row"]["sample_id"])
            assert jid not in seen          # exactly one destination each
            seen.add(jid)
    assert len(seen) == 9
    # Raw content survives byte-identically (re-serialized JSON equality).
    original = {(eid, r["sample_id"]): r for eid, rows in exports for r in rows}
    for eid, sid in seen:
        delivered = next(
            json.loads(line)["row"]
            for name in ("train", "validation", "holdout", "isolated")
            for line in (out_dir / f"{name}.jsonl").read_text(
                encoding="utf-8").splitlines()
            if (json.loads(line)["export_id"],
                json.loads(line)["row"]["sample_id"]) == (eid, sid))
        assert delivered == original[(eid, sid)]


def test_explain_chain_from_a_reaches_c_with_real_evidence(tmp_path):
    exports = hand_drawn_exports()
    _, _, out_dir, _ = write_run(tmp_path, exports, default_plan())
    ws = DistributionWorkspace.open_readonly(
        out_dir / "distribution-index.sqlite")
    try:
        result = explain_row(ws, joint_identity("exp-one", "A"))
    finally:
        ws.conn.close()
    chains = {c["window"]: c for c in result["window_chains"]}
    assert set(chains) == {"train", "validation"}
    # The asked row is itself the smallest train candidate: trivial chain.
    assert chains["train"]["hops"] == 0
    assert chains["train"]["chain_strategy"] == "trivial"
    val = chains["validation"]
    assert val["target_candidate"]["sample_id"] == "C"
    kinds = [s["edge_kind"] for s in val["steps"]]
    assert kinds == ["message", "observation"]
    # Message step: A and B, two concrete columns, one digest.
    msg_step = val["steps"][0]
    assert msg_step["from"]["sample_id"] == "A"
    assert msg_step["to"]["sample_id"] == "B"
    assert msg_step["evidence"]["message_digest"] == digest("A")
    used = msg_step["evidence"]["used_by"]
    assert used[joint_identity("exp-one", "A")]["column"] == "col0"
    assert used[joint_identity("exp-one", "B")]["column"] == "other-col"
    # Observation step: B and C with original intervals and the gap rule.
    obs_step = val["steps"][1]
    assert obs_step["from"]["sample_id"] == "B"
    assert obs_step["to"]["sample_id"] == "C"
    assert obs_step["evidence"]["gap_seconds"] == H
    assert obs_step["evidence"]["station_id"] == "ST1"
    assert obs_step["evidence"]["metric"] == "temp"


def test_explain_works_after_inputs_removed(tmp_path):
    exports = hand_drawn_exports()
    manifest_docs, _, out_dir, _ = write_run(
        tmp_path, exports, default_plan())
    index = out_dir / "distribution-index.sqlite"
    # Remove every export file the manifest points at.
    for entry in manifest_docs:
        os.remove(entry["path"])
    ws = DistributionWorkspace.open_readonly(index)
    try:
        result = explain_row(ws, joint_identity("exp-one", "A"))
    finally:
        ws.conn.close()
    assert {c["window"] for c in result["window_chains"]} == \
        {"train", "validation"}


def test_reordered_inputs_produce_identical_outputs(tmp_path):
    exports = hand_drawn_exports()
    plan = default_plan()

    def materialize(base, ordered_exports):
        d = base / "in"
        d.mkdir(parents=True)
        manifest = []
        for export_id, rows in ordered_exports:
            path = d / f"{export_id}.jsonl"
            path.write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n",
                encoding="utf-8")
            manifest.append({"export_id": export_id, "path": str(path)})
        (d / "manifest.json").write_text(json.dumps(manifest))
        (d / "plan.json").write_text(json.dumps(plan))

    base1, base2 = tmp_path / "a", tmp_path / "b"
    materialize(base1, exports)
    reversed_exports = [(eid, list(reversed(rows)))
                        for eid, rows in reversed(exports)]
    materialize(base2, reversed_exports)
    run_distribution(base1 / "in" / "manifest.json",
                     base1 / "in" / "plan.json", base1 / "out")
    run_distribution(base2 / "in" / "manifest.json",
                     base2 / "in" / "plan.json", base2 / "out")
    # Delivery rows and the per-row journal must be byte-identical: they carry
    # only export_id + content, never input paths.
    for name in ("train.jsonl", "validation.jsonl", "holdout.jsonl",
                 "isolated.jsonl", "decisions.jsonl"):
        assert (base1 / "out" / name).read_bytes() == \
            (base2 / "out" / name).read_bytes(), name
    # The summary differs only in the manifest's absolute paths.
    s1 = json.loads((base1 / "out" / "summary.json").read_text())
    s2 = json.loads((base2 / "out" / "summary.json").read_text())
    for doc in (s1, s2):
        doc.pop("exports")
    assert s1 == s2


# --------------------------------------------------------------------------- #
# relation semantics edge cases
# --------------------------------------------------------------------------- #
def test_same_package_different_message_does_not_relate(tmp_path):
    # Both rows name the same package_digest (ignored) but distinct messages.
    r1 = make_row("P", "ST1", "temp", 1, 2, digests=("P",))
    r2 = make_row("Q", "ST2", "temp", 13, 14, digests=("Q",))
    for row in (r1, r2):
        for f in row["features"]:
            f["evidence"]["field"]["package_digest"] = "same-package"
    exports = [("e1", [r1, r2])]
    _, _, out_dir, report = write_run(tmp_path, exports, default_plan())
    assert report.group_count == 2
    decisions = load_decisions(out_dir)
    assert decisions[("e1", "P")]["decision"] == "train"
    assert decisions[("e1", "Q")]["decision"] == "holdout"


def test_same_message_in_different_packages_relates(tmp_path):
    r1 = make_row("P", "ST1", "temp", 1, 2, digests=("Z",))
    r2 = make_row("Q", "ST2", "temp", 13, 14, digests=("Z",))
    r1["features"][0]["evidence"]["field"]["package_digest"] = "pkg-1"
    r2["features"][0]["evidence"]["field"]["package_digest"] = "pkg-2"
    exports = [("e1", [r1]), ("e2", [r2])]
    _, _, out_dir, _ = write_run(tmp_path, exports, default_plan())
    decisions = load_decisions(out_dir)
    assert decisions[("e1", "P")]["decision"] == "isolated"
    assert decisions[("e2", "Q")]["decision"] == "isolated"
    assert decisions[("e1", "P")]["group"] == decisions[("e2", "Q")]["group"]


def test_repeated_digest_within_one_row_stays_linear(tmp_path):
    # Y cites one digest in two columns; the star builder emits one edge per
    # non-hub *reference*, so k references still mean k-1 edges, never a
    # quadratic table, and both rows land in one component.
    r1 = make_row("X", "ST7", "temp", 1, 2, digests=("Z",))
    r2 = make_row("Y", "ST8", "temp", 15, 16, digests=("Z", "Z"),
                  columns=("q", "r"))
    _, _, out_dir, report = write_run(
        tmp_path, [("e1", [r1, r2])], default_plan())
    assert report.group_count == 1
    scale = report.relation_scale
    assert scale["successful_feature_refs"] == 3
    assert scale["message_edges"] == 2  # k=3 references -> k-1 star edges
    decisions = load_decisions(out_dir)
    assert decisions[("e1", "X")]["group"] == decisions[("e1", "Y")]["group"]


def test_equal_values_alone_do_not_relate(tmp_path):
    r1 = make_row("P", "ST1", "temp", 1, 2, digests=("P",))
    r2 = make_row("Q", "ST2", "temp", 13, 14, digests=("Q",))
    # Identical values, different digests, different station: unrelated.
    r1["features"][0]["value"] = 42.0
    r2["features"][0]["value"] = 42.0
    exports = [("e1", [r1, r2])]
    _, _, out_dir, report = write_run(tmp_path, exports, default_plan())
    assert report.group_count == 2


def test_digest_on_failed_feature_is_ignored(tmp_path):
    r1 = make_row("P", "ST1", "temp", 1, 2)
    r2 = make_row("Q", "ST-OTHER", "temp", 13, 14)
    for row, letter in ((r1, "Z"), (r2, "Z")):
        row["features"][0]["status"] = "field_missing"
        row["features"][0]["value"] = None
        row["features"][0]["evidence"]["field"]["message_digest"] = \
            digest(letter)
    exports = [("e1", [r1, r2])]
    _, _, out_dir, report = write_run(tmp_path, exports, default_plan())
    # Same digest retained by missing features must not create a relation,
    # and the distinct stations rule out an observation relation.
    assert report.group_count == 2


def test_exact_gap_does_not_relate_but_smaller_gap_does(tmp_path):
    # C [4,6), D [7,8): empty gap exactly gap_seconds -> unrelated.
    c = make_row("C", "ST1", "temp", 4, 6, digests=("C",))
    d = make_row("D", "ST1", "temp", 7, 8, digests=("D",))
    _, _, out_dir, report = write_run(
        tmp_path, [("e1", [c, d])], default_plan(gap_seconds=H))
    assert report.group_count == 2

    # Same geometry with a larger configured gap -> related.
    base2 = tmp_path / "run2"
    base2.mkdir()
    _, _, out2, report2 = write_run(
        base2, [("e1", [json.loads(json.dumps(c)),
                        json.loads(json.dumps(d))])],
        default_plan(gap_seconds=H + 1))
    assert report2.group_count == 1


def test_gap_zero_touching_intervals_do_not_relate(tmp_path):
    r1 = make_row("P", "ST1", "temp", 1, 2, digests=("P",))
    r2 = make_row("Q", "ST1", "temp", 2, 3, digests=("Q",))  # touches r1
    _, _, out_dir, report = write_run(
        tmp_path, [("e1", [r1, r2])], default_plan(gap_seconds=0))
    assert report.group_count == 2


def test_touching_row_remains_a_bridge_to_overlapping_rows(tmp_path):
    # B and C are separated by exactly gap_seconds and never get a direct
    # edge, but B2 touches B while its provisional interval still reaches C.
    # Connectivity must flow A-B-B2-C (transitivity via the touching row).
    rows = [
        make_row("A", "ST1", "temp", 1, 3, digests=("a",)),
        make_row("B", "ST1", "temp", 3, 5, digests=("b",)),
        make_row("B2", "ST1", "temp", 5, 6, digests=("c",)),
        make_row("C", "ST1", "temp", 6, 8, digests=("d",)),
    ]
    _, _, out_dir, report = write_run(
        tmp_path, [("e1", rows)], default_plan(gap_seconds=H))
    assert report.group_count == 1
    ws = DistributionWorkspace.open_readonly(
        out_dir / "distribution-index.sqlite")
    try:
        pairs = {
            tuple(sorted((r["a_id"].split("\x1f")[1],
                          r["b_id"].split("\x1f")[1])))
            for r in ws.conn.execute(
                "SELECT a_id, b_id FROM edges "
                "WHERE edge_kind = 'observation'")}
    finally:
        ws.conn.close()
    # No direct B-C edge (exact gap), but the bridge chain is present.
    assert ("B", "C") not in pairs
    assert ("A", "B") in pairs and ("B", "B2") in pairs
    assert ("B2", "C") in pairs


def test_nested_long_interval_connects_short_ones(tmp_path):
    # One long interval L [1,9) contains short S1 [2,3) and S2 [5,6); the
    # shorts are disjoint but both overlap L, so all three are one component.
    rows = [
        make_row("L", "ST1", "temp", 1, 9, digests=("L",)),
        make_row("S1", "ST1", "temp", 2, 3, digests=("S1",)),
        make_row("S2", "ST1", "temp", 5, 6, digests=("S2",)),
    ]
    exports = [("e1", rows)]
    plan = default_plan(gap_seconds=0)
    _, _, out_dir, report = write_run(tmp_path, exports, plan)
    assert report.group_count == 1
    judge = NaiveJudge(exports, plan)
    decisions = load_decisions(out_dir)
    for jid, (decision, reasons, group_min) in judge.expected().items():
        key = tuple(jid.split("\x1f", 1))
        assert decisions[key]["decision"] == decision
        assert sorted(decisions[key]["reasons"]) == sorted(reasons)


def test_label_revision_and_value_are_ignored_in_obs_relation(tmp_path):
    # Same station/metric/intervals relate regardless of label status/value.
    r1 = make_row("P", "ST1", "temp", 1, 2, label="ok", digests=("P",))
    r2 = make_row("Q", "ST1", "temp", 1, 2, label="label_retracted",
                  digests=("Q",))
    r2["label"]["winning_record"] = {"seq": 99}
    exports = [("e1", [r1, r2])]
    _, _, out_dir, report = write_run(tmp_path, exports, default_plan())
    assert report.group_count == 1


def test_identical_sample_ids_in_different_exports_are_not_merged(tmp_path):
    r1 = make_row("SAME", "ST1", "temp", 1, 2, digests=("P",))
    r2 = make_row("SAME", "ST2", "temp", 13, 14, digests=("Q",))
    exports = [("e1", [r1]), ("e2", [r2])]
    _, _, out_dir, report = write_run(tmp_path, exports, default_plan())
    assert report.row_count == 2
    decisions = load_decisions(out_dir)
    assert ("e1", "SAME") in decisions and ("e2", "SAME") in decisions
    assert decisions[("e1", "SAME")]["group"] != \
        decisions[("e2", "SAME")]["group"]


def test_three_window_group_isolates_all_candidates(tmp_path):
    # One digest shared by a candidate in each of the three windows.
    rows = [
        make_row("T", "ST1", "temp", 1, 2, digests=("Z",)),
        make_row("V", "ST2", "temp", 5, 6, digests=("Z",)),
        make_row("G", "ST3", "temp", 13, 14, digests=("Z",)),
    ]
    exports = [("e1", rows)]
    _, _, out_dir, _ = write_run(tmp_path, exports, default_plan())
    decisions = load_decisions(out_dir)
    for sid in ("T", "V", "G"):
        assert decisions[("e1", sid)]["decision"] == "isolated"
        assert "group_spans_multiple_windows" in \
            decisions[("e1", sid)]["reasons"]


def test_multiple_direct_reasons_are_all_retained(tmp_path):
    # Crosses a window boundary AND has no usable label.
    row = make_row("E", "ST5", "temp", 3, 12, label="label_retracted",
                   digests=("E",))
    _, _, out_dir, _ = write_run(
        tmp_path, [("e1", [row])], default_plan())
    decision = load_decisions(out_dir)[("e1", "E")]
    assert sorted(decision["reasons"]) == sorted(
        ["crosses_windows", "label_unavailable"])


# --------------------------------------------------------------------------- #
# hard errors: reject, never treat missing evidence as unrelated
# --------------------------------------------------------------------------- #
def _single_export(tmp_path, rows, plan=None):
    path = tmp_path / "e.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n",
                    encoding="utf-8")
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps(
        [{"export_id": "e1", "path": str(path)}]))
    plan_path = tmp_path / "p.json"
    plan_path.write_text(json.dumps(plan or default_plan()))
    return manifest, plan_path


def test_duplicate_sample_id_within_export_rejected(tmp_path):
    rows = [make_row("DUP", "ST1", "temp", 1, 2),
            make_row("DUP", "ST2", "temp", 13, 14)]
    manifest, plan_path = _single_export(tmp_path, rows)
    out_dir = tmp_path / "out"
    with pytest.raises(DistributionError) as exc:
        run_distribution(manifest, plan_path, out_dir)
    assert exc.value.reason == "duplicate_joint_identity"
    # A failed run never publishes any delivery artifact.
    assert not any(out_dir.glob("*")) if out_dir.exists() else True


def test_bad_interval_rejected(tmp_path):
    row = make_row("X", "ST1", "temp", 2, 2)
    manifest, plan_path = _single_export(tmp_path, [row])
    with pytest.raises(DistributionError) as exc:
        run_distribution(manifest, plan_path, tmp_path / "out")
    assert exc.value.reason == "row_interval_invalid"


def test_successful_feature_without_digest_rejected(tmp_path):
    row = make_row("X", "ST1", "temp", 1, 2)
    del row["features"][0]["evidence"]["field"]["message_digest"]
    manifest, plan_path = _single_export(tmp_path, [row])
    with pytest.raises(DistributionError) as exc:
        run_distribution(manifest, plan_path, tmp_path / "out")
    assert exc.value.reason == \
        "successful_feature_without_message_digest"


def test_illegal_digest_shape_rejected(tmp_path):
    row = make_row("X", "ST1", "temp", 1, 2)
    row["features"][0]["evidence"]["field"]["message_digest"] = "nope"
    manifest, plan_path = _single_export(tmp_path, [row])
    with pytest.raises(DistributionError) as exc:
        run_distribution(manifest, plan_path, tmp_path / "out")
    assert exc.value.reason == \
        "successful_feature_without_message_digest"


def test_overlapping_windows_rejected(tmp_path):
    row = make_row("X", "ST1", "temp", 1, 2)
    plan = default_plan()
    plan["windows"]["validation"]["start"] = "2026-01-01T03:30:00Z"
    manifest, plan_path = _single_export(tmp_path, [row], plan)
    with pytest.raises(DistributionError) as exc:
        run_distribution(manifest, plan_path, tmp_path / "out")
    assert exc.value.reason == "plan_windows_overlap"


def test_duplicate_export_id_rejected(tmp_path):
    p1 = tmp_path / "a.jsonl"
    p2 = tmp_path / "b.jsonl"
    p1.write_text(json.dumps(make_row("A", "ST1", "temp", 1, 2)) + "\n")
    p2.write_text(json.dumps(make_row("B", "ST2", "temp", 13, 14)) + "\n")
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps([
        {"export_id": "dup", "path": str(p1)},
        {"export_id": "dup", "path": str(p2)}]))
    (tmp_path / "p.json").write_text(json.dumps(default_plan()))
    with pytest.raises(DistributionError) as exc:
        run_distribution(manifest, tmp_path / "p.json", tmp_path / "out")
    assert exc.value.reason == "duplicate_export_id"


def test_missing_export_file_rejected(tmp_path):
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps(
        [{"export_id": "e1", "path": str(tmp_path / "gone.jsonl")}]))
    (tmp_path / "p.json").write_text(json.dumps(default_plan()))
    with pytest.raises(DistributionError) as exc:
        run_distribution(manifest, tmp_path / "p.json", tmp_path / "out")
    assert exc.value.reason == "manifest_file_missing"


# --------------------------------------------------------------------------- #
# linear edge bounds and persistence
# --------------------------------------------------------------------------- #
def test_edge_counts_are_linear_and_index_is_self_contained(tmp_path):
    exports = hand_drawn_exports()
    _, _, out_dir, report = write_run(tmp_path, exports, default_plan())
    scale = report.relation_scale
    n_rows = scale["rows"]
    n_refs = scale["successful_feature_refs"]
    assert scale["message_edges"] <= n_refs
    assert scale["observation_edges"] < n_rows
    assert scale["total_edges"] <= n_rows + n_refs
    assert scale["forest_edges"] <= n_rows - 1

    # The delivered index alone answers every edge question.
    ws = DistributionWorkspace.open_readonly(
        out_dir / "distribution-index.sqlite")
    try:
        stored_edges = ws.conn.execute(
            "SELECT COUNT(*) c FROM edges").fetchone()["c"]
        stored_refs = ws.conn.execute(
            "SELECT COUNT(*) c FROM message_refs").fetchone()["c"]
        assert stored_edges == scale["total_edges"]
        assert stored_refs == n_refs
        # Every observation edge actually satisfies the positive-overlap rule.
        gap = default_plan()["gap_seconds"]
        for edge in ws.conn.execute(
                "SELECT * FROM edges WHERE edge_kind = 'observation'"):
            assert edge["b_start"] < edge["a_end"] + gap
            assert edge["a_start"] < edge["b_end"] + gap
            assert (edge["b_start"] - edge["a_end"]) < gap
        # No message edge exists for a failed feature.
        assert ws.conn.execute(
            "SELECT COUNT(*) c FROM edges e WHERE e.edge_kind = 'message' "
            "AND NOT EXISTS (SELECT 1 FROM message_refs r "
            "WHERE r.joint_id = e.a_id AND r.digest = e.digest)").fetchone()[
            "c"] == 0
    finally:
        ws.conn.close()


def test_deterministic_chain_under_bfs_budget(tmp_path):
    exports = hand_drawn_exports()
    _, _, out_dir, _ = write_run(tmp_path, exports, default_plan())
    ws = DistributionWorkspace.open_readonly(
        out_dir / "distribution-index.sqlite")
    try:
        full = explain_row(ws, joint_identity("exp-one", "A"),
                           max_chain_nodes=5000)
        tiny = explain_row(ws, joint_identity("exp-one", "A"),
                           max_chain_nodes=2)
    finally:
        ws.conn.close()
    # The tiny budget forces the spanning-forest fallback; both must still
    # reach the same validation candidate with valid evidence.
    full_val = next(c for c in full["window_chains"]
                    if c["window"] == "validation")
    tiny_val = next(c for c in tiny["window_chains"]
                    if c["window"] == "validation")
    assert tiny_val["chain_strategy"] == "dsu_forest_fallback"
    assert full_val["target_candidate"] == tiny_val["target_candidate"]
    for step in tiny_val["steps"]:
        assert step["edge_kind"] in ("message", "observation")


def test_cli_distribute_and_explain(tmp_path):
    exports = hand_drawn_exports()
    manifest, plan_path, out_dir, _ = write_run(
        tmp_path, exports, default_plan())
    runner = tmp_path / "run_cli.py"
    runner.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from grib_asm.training.cli import main\n"
        "raise SystemExit(main())\n", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-I", str(runner), "cube-training-explain",
         "--index", str(out_dir / "distribution-index.sqlite"),
         "--export-id", "exp-one", "--sample-id", "A"],
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert {c["window"] for c in payload["window_chains"]} == \
        {"train", "validation"}


def test_cli_error_exit_codes(tmp_path):
    runner = tmp_path / "run_cli.py"
    runner.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from grib_asm.training.cli import main\n"
        "raise SystemExit(main())\n", encoding="utf-8")
    good_plan = default_plan()
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(good_plan))
    # A hard distribution error exits 7 with a structured reason.
    manifest_path = tmp_path / "missing.json"
    manifest_path.write_text(json.dumps(
        [{"export_id": "x", "path": str(tmp_path / "gone.jsonl")}]))
    proc = subprocess.run(
        [sys.executable, "-I", str(runner), "cube-training-distribute",
         "--manifest", str(manifest_path), "--plan", str(plan_path),
         "--out-dir", str(tmp_path / "o")],
        capture_output=True, text=True)
    assert proc.returncode == 7
    assert json.loads(proc.stdout)["error"] == "manifest_file_missing"
    # A missing index is a usage/path failure, exit 2 (no traceback).
    proc = subprocess.run(
        [sys.executable, "-I", str(runner), "cube-training-explain",
         "--index", str(tmp_path / "no-index.sqlite"),
         "--export-id", "x", "--sample-id", "y"],
        capture_output=True, text=True)
    assert proc.returncode == 2
    assert "distribution index not found" in proc.stderr


def test_small_page_limit_matches_judge(tmp_path):
    # Force keyset pagination paths with a deliberately tiny page/batch size.
    config = DistributionConfig(page_limit=2, commit_every=1,
                                sqlite_cache_kib=512)
    exports = hand_drawn_exports()
    plan = default_plan()
    _, _, out_dir, _ = write_run(tmp_path, exports, plan, config=config)
    decisions = load_decisions(out_dir)
    expected = NaiveJudge(exports, plan).expected()
    for jid, (decision, reasons, _group) in expected.items():
        key = tuple(jid.split("\x1f", 1))
        assert decisions[key]["decision"] == decision
        assert sorted(decisions[key]["reasons"]) == sorted(reasons)
