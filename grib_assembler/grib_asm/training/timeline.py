"""Point-in-time resolution of the visible history for one business key.

These functions encode the single ordering rule shared by all three streams:

* records are ordered by their UTC visibility time; on a tie the record with
  the larger non-negative sequence number wins - line order in the JSONL file
  is never used to adjudicate;
* a resend of an *identical* record (same business key, same visibility time,
  same sequence, same full payload) is deduplicated;
* the same visibility time and sequence number carrying different payloads is
  an input conflict and raises, rather than being resolved by file order.

The two clocks appear as the ``cutoff_epoch`` argument: only records visible
at or before the cutoff participate (equality counts as visible).  Features
call these resolvers with a sample's ``decision_at``; labels call them with
the recipe's ``label_cutoff``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .errors import StreamError
from .model import StreamRecord, epoch_iso


@dataclass(frozen=True)
class VisibleRecord:
    """The winning record of one business key at one clock cutoff."""

    record_id: str
    visible_at_epoch: int
    seq: int
    action: str
    doc: dict

    @property
    def visible_at(self) -> str:
        return epoch_iso(self.visible_at_epoch)

    def identity(self) -> dict:
        """Audit identity attached to every output cell."""

        return {
            "record_id": self.record_id,
            "visible_at": self.visible_at,
            "seq": self.seq,
        }


def assert_no_same_instant_conflict(records: list[StreamRecord]) -> None:
    """Fail on (business key, visible_at, seq) collisions with differing body.

    Runs once while a stream is ingested, so resolution queries never have to
    re-scan history.  Identical retransmissions are silently acceptable.
    """

    seen: dict[tuple[str, int, int], str] = {}
    for rec in records:
        token = (rec.business_key, rec.visible_at_epoch, rec.seq)
        previous = seen.get(token)
        if previous is None:
            seen[token] = rec.payload
        elif previous != rec.payload:
            raise StreamError("stream_duplicate_conflict", {
                "stream": rec.stream,
                "business_key": json.loads(rec.business_key),
                "visible_at": epoch_iso(rec.visible_at_epoch),
                "seq": rec.seq,
                "record_ids": [
                    rid for rid in _record_ids(records, token)],
                "reason": "same visibility time and sequence but different "
                          "content; input order cannot adjudicate"})


def _record_ids(records, token):
    for rec in records:
        if (rec.business_key, rec.visible_at_epoch, rec.seq) == token:
            yield rec.record_id


def latest_visible(records: list[StreamRecord],
                   cutoff_epoch: int) -> VisibleRecord | None:
    """Return the newest record visible at ``cutoff_epoch`` or ``None``.

    ``records`` holds the full history of one business key.  Records at the
    same instant are already conflict-checked; identical payloads collapse to
    one.  Ties therefore never reach the comparison.
    """

    winner: StreamRecord | None = None
    for rec in records:
        if rec.visible_at_epoch > cutoff_epoch:
            continue
        if winner is None or (
                rec.visible_at_epoch > winner.visible_at_epoch or
                (rec.visible_at_epoch == winner.visible_at_epoch
                 and rec.seq > winner.seq)):
            winner = rec
    if winner is None:
        return None
    return VisibleRecord(winner.record_id, winner.visible_at_epoch,
                         winner.seq, winner.action, winner.doc)
