"""Reconstruct why an isolated row cannot be distributed.

The distribution index (``distribution-index.sqlite``) is self-contained: it
keeps the rows' typed facts, every successful-feature message reference and the
union forest's evidence-bearing edges.  Explanations therefore work *after the
input export files have been removed* and never rescan the data set.

For an asked row whose connected group contains candidates of two or three
different delivery windows, :func:`explain_row` returns, for every such window,
a stable association chain from the asked row to that window's smallest-id
candidate:

* a **message** step names both endpoint rows, the two concrete feature
  columns that actually used one shared ``message_digest`` and the digest
  itself (re-checkable against ``message_refs``);
* an **observation** step names both endpoint rows, their original
  ``(station_id, metric)`` target intervals and the ``gap_seconds`` rule that
  made the extended intervals overlap with positive length.

The forest is built with union-by-size and deterministic tie-breaking, so the
same input always yields the same roots, edges and chains; each hop is a
primary-key edge lookup along a logarithmic root path, never a table scan.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .distribute_errors import DistributionError
from .distribute_rows import split_joint_identity
from .distribute_workspace import EDGE_MESSAGE
from .model import epoch_iso


@dataclass
class _Hop:
    edge_id: int
    lower: str       # the node closer to the asked row, chain direction
    higher: str


def _identity_doc(joint_id: str) -> dict:
    export_id, sample_id = split_joint_identity(joint_id)
    return {"export_id": export_id, "sample_id": sample_id,
            "joint_id": joint_id}


def _other_end(edge, node: str) -> str:
    return edge["b_id"] if edge["a_id"] == node else edge["a_id"]


def _forest_hops(workspace, asked: str, target: str) -> list[_Hop]:
    """Logarithmic path through the union-by-size spanning forest.

    Used only when component-local BFS exceeds its resident-node budget; every
    hop is still a real stored edge with re-checkable evidence, and the path
    is fully deterministic because forest construction is order-independent.
    """

    path_a = workspace.ancestors(asked)
    path_t = workspace.ancestors(target)
    ancestors_of_target = set(path_t)
    lca = next(node for node in path_a if node in ancestors_of_target)

    edge_of: dict[str, int] = {}
    for path in (path_a, path_t):
        for node in path[:-1]:
            edge_of[node] = int(
                workspace.get_row(node)["parent_edge_id"])

    hops: list[_Hop] = []
    for node in path_a:                      # asked -> ... -> LCA
        if node == lca:
            break
        edge = workspace.get_edge(edge_of[node])
        hops.append(_Hop(edge_of[node], lower=node,
                         higher=_other_end(edge, node)))
    for node in reversed(path_t[:path_t.index(lca)]):  # LCA -> ... -> target
        edge_id = edge_of[node]
        edge = workspace.get_edge(edge_id)
        hops.append(_Hop(edge_id, lower=_other_end(edge, node),
                         higher=node))
    return hops


def chain_hops(workspace, asked: str, target: str,
               max_chain_nodes: int) -> tuple[list[_Hop], str, int]:
    """Return ``(hops, strategy, nodes_visited)`` for an ordered, evidence path.

    The primary strategy is deterministic BFS over edges incident on reached
    nodes - indexed lookups only, never a whole-table scan.  Resident state is
    capped at ``max_chain_nodes`` joint identities: if the target has not been
    reached within that budget the search aborts and the logarithmic DSU
    spanning-forest path is used instead, so a million-row hub can never force
    the explanation process to hold a million nodes in memory.  Both strategies
    produce identical output on every rerun of the same index.
    """

    if asked == target:
        return [], "trivial", 1
    predecessors: dict[str, tuple[str, int]] = {}
    frontier = [asked]
    seen = {asked}
    head = 0
    reached = False
    while head < len(frontier):
        node = frontier[head]
        head += 1
        for edge in workspace.neighbors(node):
            other = _other_end(edge, node)
            if other in seen:
                continue
            if len(seen) >= max_chain_nodes:
                hops = _forest_hops(workspace, asked, target)
                return hops, "dsu_forest_fallback", len(seen)
            seen.add(other)
            predecessors[other] = (node, int(edge["edge_id"]))
            if other == target:
                reached = True
                frontier.clear()
                break
            frontier.append(other)
    if not reached:
        raise DistributionError("explain_chain_broken", {
            "asked": asked, "target": target,
            "reason": "both rows share a component but no stored edge path "
                      "connects them - the index is inconsistent"})
    reversed_hops: list[_Hop] = []
    node = target
    while node != asked:
        previous, edge_id = predecessors[node]
        reversed_hops.append(_Hop(edge_id, lower=previous, higher=node))
        node = previous
    reversed_hops.reverse()
    return reversed_hops, "edge_bfs", len(seen)


def _render_step(workspace, hop: _Hop) -> dict:
    edge = workspace.get_edge(hop.edge_id)
    step = {
        "edge_id": hop.edge_id,
        "edge_kind": edge["edge_kind"],
        "from": _identity_doc(hop.lower),
        "to": _identity_doc(hop.higher),
    }
    if edge["edge_kind"] == EDGE_MESSAGE:
        # Edges are stored with the hub in ``a_id`` (column_a) and the other
        # endpoint in ``b_id`` (column_b); BFS may traverse them in either
        # direction, so map each rendered endpoint back to its own column.
        digest = edge["digest"]
        column_of = {edge["a_id"]: edge["column_a"],
                     edge["b_id"]: edge["column_b"]}
        for endpoint in (hop.lower, hop.higher):
            column = column_of[endpoint]
            found = workspace.conn.execute(
                "SELECT 1 FROM message_refs WHERE joint_id = ? AND digest = ? "
                "AND column_name = ? LIMIT 1",
                (endpoint, digest, column)).fetchone()
            if found is None:
                raise DistributionError("explain_evidence_inconsistent", {
                    "edge_id": hop.edge_id, "joint_id": endpoint,
                    "digest": digest, "column": column})
        step["evidence"] = {
            "rule": "both rows consumed the same original GRIB2 message; "
                    "equal values or shared packages alone never relate rows",
            "message_digest": digest,
            "used_by": {
                hop.lower: {"column": column_of[hop.lower]},
                hop.higher: {"column": column_of[hop.higher]}}}
    else:
        gap_seconds = int(edge["gap_seconds"])
        # Interval evidence is keyed by the actual endpoint joint ids, so the
        # rendering stays correct under reverse traversal as well.
        interval_of = {
            edge["a_id"]: [int(edge["a_start"]), int(edge["a_end"])],
            edge["b_id"]: [int(edge["b_start"]), int(edge["b_end"])]}
        first_id, second_id = hop.lower, hop.higher
        first_interval = interval_of[first_id]
        second_interval = interval_of[second_id]
        # Positive when an empty space separates the intervals, zero when they
        # touch, negative when the second starts inside the first.
        interval_gap = second_interval[0] - first_interval[1]
        step["evidence"] = {
            "rule": "same station_id and metric; each [start,end) is treated "
                    "as [start,end+gap_seconds); the extended intervals share "
                    "a positive-length section, so the rows relate. An exact "
                    "gap of gap_seconds and touching intervals do not relate",
            "station_id": edge["station_id"],
            "metric": edge["metric"],
            "gap_seconds": gap_seconds,
            "intervals": {
                first_id: {
                    "target_interval_epoch": first_interval,
                    "target_interval": [epoch_iso(first_interval[0]),
                                        epoch_iso(first_interval[1])],
                    "extended_end_epoch": first_interval[1] + gap_seconds},
                second_id: {
                    "target_interval_epoch": second_interval,
                    "target_interval": [epoch_iso(second_interval[0]),
                                        epoch_iso(second_interval[1])],
                    "extended_end_epoch": second_interval[1] + gap_seconds}},
            "empty_gap_between_intervals_seconds": interval_gap,
            "related_because": (
                f"the second interval starts {interval_gap} seconds after the "
                f"first ends, which is < gap_seconds={gap_seconds}; the "
                "provisional extended intervals therefore overlap with "
                "positive length")}
    return step


def explain_row(workspace, asked_joint_id: str,
                max_chain_nodes: int = 5000) -> dict:
    """Build the cross-window explanation for one isolated joint identity.

    ``max_chain_nodes`` bounds the resident BFS state per window chain; once
    exceeded the logarithmic spanning-forest path is used instead.
    """

    asked = workspace.get_row(asked_joint_id)
    if asked is None:
        raise DistributionError("explain_unknown_identity",
                                {"joint_id": asked_joint_id})
    group_id = asked["group_id"]
    # One smallest-id candidate per window present in this group: deterministic
    # without scanning the whole table (indexed group/window lookups).
    per_window: dict[str, object] = {}
    for window in ("train", "validation", "holdout"):
        cand = workspace.conn.execute(
            "SELECT joint_id FROM rows WHERE group_id = ? AND candidate = 1 "
            "AND window_name = ? ORDER BY joint_id LIMIT 1",
            (group_id, window)).fetchone()
        if cand is not None:
            per_window[window] = cand["joint_id"]
    if len(per_window) < 2:
        raise DistributionError("explain_not_cross_window_isolated", {
            "joint_id": asked_joint_id,
            "decision": asked["destination"],
            "reasons": json.loads(asked["reasons_json"]),
            "group_windows_present": sorted(per_window),
            "reason": "the connected group has candidates in fewer than two "
                      "delivery windows, so this row was not isolated for "
                      "cross-window source sharing"})

    explanations = []
    for window in sorted(per_window):
        target = per_window[window]
        if target == asked_joint_id:
            chain = {"window": window,
                     "target_candidate": _identity_doc(target),
                     "hops": 0,
                     "chain_strategy": "trivial",
                     "nodes_visited": 1,
                     "steps": [],
                     "note": "the asked row is itself this window's candidate"}
        else:
            hops, strategy, visited = chain_hops(
                workspace, asked_joint_id, target, max_chain_nodes)
            chain = {
                "window": window,
                "target_candidate": _identity_doc(target),
                "hops": len(hops),
                "chain_strategy": strategy,
                "nodes_visited": visited,
                "steps": [_render_step(workspace, hop) for hop in hops]}
        explanations.append(chain)

    return {
        "asked": _identity_doc(asked_joint_id),
        "asked_decision": asked["destination"],
        "asked_reasons": json.loads(asked["reasons_json"]),
        "asked_window": asked["window_name"],
        "group": _identity_doc(group_id),
        "conclusion": "the connected group contains label-ok candidates in "
                      f"{len(per_window)} different windows "
                      f"({', '.join(sorted(per_window))}); every candidate of "
                      "the group must be isolated - no window may be chosen "
                      "by majority, first match or priority",
        "window_chains": explanations,
    }
