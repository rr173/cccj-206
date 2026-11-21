"""Randomized property tests: engine versus an independent brute-force judge.

Each iteration generates a small random world (random stations, metrics,
intervals, digests, labels, two exports and a randomly sized gap) and compares
every product decision, reason set and group against ``NaiveJudge``, which
enumerates all related pairs and computes transitive closure itself.  The
random seed makes failures reproducible while the configurations themselves
are not derived from product code.
"""

from __future__ import annotations

import hashlib
import random
from datetime import datetime, timezone

import pytest

from grib_asm.training.distribute import DistributionConfig
from grib_asm.training.distribute_rows import joint_identity

from test_training_distribute import (
    NaiveJudge, T0, H, load_decisions, write_run,
)


def _digest(seed: int) -> str:
    return hashlib.sha256(f"msg-{seed}".encode()).hexdigest()


def _iso(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def _plan(gap_seconds: int):
    return {
        "gap_seconds": gap_seconds,
        "windows": {
            "train": {"start": _iso(T0 + 1 * H), "end": _iso(T0 + 4 * H)},
            "validation": {"start": _iso(T0 + 4 * H),
                           "end": _iso(T0 + 10 * H)},
            "holdout": {"start": _iso(T0 + 12 * H),
                        "end": _iso(T0 + 18 * H)}}}


def _random_world(rng: random.Random):
    n_rows = rng.randint(4, 24)
    stations = [f"ST{i}" for i in range(rng.randint(1, 5))]
    metrics = ["temp", "rain"]
    # A small digest pool forces frequent shared-message components.
    digest_pool = [_digest(i) for i in range(rng.randint(1, 6))]
    rows = []
    for i in range(n_rows):
        start_h = rng.randint(0, 20)
        end_h = start_h + rng.randint(1, 4)
        n_features = rng.randint(1, 2)
        features = []
        for f in range(n_features):
            status = "ok" if rng.random() < 0.8 else rng.choice(
                ("field_missing", "forecast_withdrawn", "point_missing"))
            feature = {
                "column": f"c{f}", "status": status,
                "value": 1.0 if status == "ok" else None,
                "evidence": {"field": {
                    "message_digest": rng.choice(digest_pool),
                    "package_digest": rng.choice(("pkg1", "pkg2"))}}}
            features.append(feature)
        label_status = "ok" if rng.random() < 0.75 else rng.choice(
            ("label_not_received", "label_retracted",
             "label_unit_mismatch"))
        rows.append({
            "sample_id": f"S{i:03d}",
            "station_id": rng.choice(stations),
            "metric": rng.choice(metrics),
            "target_interval": [T0 + start_h * H, T0 + end_h * H],
            "decision_at": _iso(T0),
            "features": features,
            "label": {"status": label_status,
                      "value": 1.0 if label_status == "ok" else None,
                      "unit": "K",
                      "revision": rng.randint(0, 5)}})
    rng.shuffle(rows)
    cut = rng.randint(1, len(rows) - 1)
    # Same sample_id may appear in both exports (distinct joint identities);
    # within one export ids stay unique.
    exports = [("exp-a", rows[:cut]), ("exp-b", rows[cut:])]
    return exports


@pytest.mark.parametrize("seed", range(40))
def test_random_worlds_match_naive_judge(tmp_path, seed):
    rng = random.Random(1000 + seed)
    exports = _random_world(rng)
    gap = rng.choice((0, 1, 600, H, 2 * H))
    plan = _plan(gap)
    config = DistributionConfig(
        page_limit=rng.choice((2, 5, 2000)),
        commit_every=rng.choice((1, 100, 2000)),
        sqlite_cache_kib=rng.choice((256, 16 * 1024)))
    _, _, out_dir, report = write_run(tmp_path, exports, plan,
                                      config=config)
    decisions = load_decisions(out_dir)
    judge = NaiveJudge(exports, plan)
    expected = judge.expected()
    assert len(decisions) == len(expected)
    for jid, (decision, reasons, group_min) in expected.items():
        key = tuple(jid.split("\x1f", 1))
        doc = decisions[key]
        assert doc["decision"] == decision, (seed, jid)
        assert sorted(doc["reasons"]) == sorted(reasons), (seed, jid)
        assert tuple(joint_identity(doc["group"]["export_id"],
                                    doc["group"]["sample_id"]).split("\x1f",
                                                                     1)) \
            == tuple(group_min.split("\x1f", 1)), (seed, jid)
    # Conservation holds on every random world.
    assert sum(report.shard_counts.values()) + report.isolated_count \
        == report.row_count
